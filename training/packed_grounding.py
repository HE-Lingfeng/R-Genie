"""Memory-map verified batches of frozen spatial-head inputs across training epochs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from training.paper_utils import sha256

CHUNK_SIZE = 32


class PackedInputs:
    def __init__(self, dataset, directory, layout):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.size = len(dataset)
        spec = dict(partition=dataset.identity, rows=self.size, chunk_size=CHUNK_SIZE, layout=layout)
        # Names based on content let persistent workers finish an older run undisturbed.
        key = hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
        self.folder = directory/key
        with (directory/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self.folder.is_dir():
                self._create(dataset,spec)
        manifest = json.loads((self.folder/'index.json').read_bytes())
        parts = manifest['chunks']
        expected = [self._name(start//CHUNK_SIZE) for start in range(0,self.size,CHUNK_SIZE)]
        if manifest['input']!=spec or list(parts)!=expected:
            raise ValueError('The packed dataset differs from its canonical feature receipts')
        for name in expected:
            if sha256(self.folder/name)!=parts[name]:
                raise ValueError(f'Packed feature chunk changed: {name}')
        self._process = None
        self._mapped = {}

    @staticmethod
    def _name(index):
        return f'{index:05d}.pt'

    def _create(self, dataset, spec):
        with TemporaryDirectory(dir=self.folder.parent) as scratch:
            # Make a child that can be atomically renamed without moving the temporary directory itself.
            publish = Path(scratch)/'pack'
            publish.mkdir()
            chunks = {}
            for start in range(0,self.size,CHUNK_SIZE):
                # The normal dataset path checks the per-image digest, source binding and tensor contents.
                inputs = [dataset[i][0] for i in range(start,min(start+CHUNK_SIZE,self.size))]
                tensors = tuple(torch.stack(field) for field in zip(*inputs,strict=True))
                name = self._name(start//CHUNK_SIZE)
                torch.save(tensors,publish/name)
                chunks[name] = sha256(publish/name)
            (publish/'index.json').write_text(json.dumps(dict(input=spec,chunks=chunks)))
            publish.rename(self.folder)

    def __getstate__(self):
        # Spawned workers reopen read-only inputs instead of serializing their parent's mapped storages.
        return {**self.__dict__,'_process':None,'_mapped':{}}

    def __getitem__(self, index):
        if index<0:
            index+=self.size
        if not 0<=index<self.size:
            raise IndexError(index)
        pid = os.getpid()
        if self._process!=pid:
            self._mapped={}
            self._process=pid
        chunk,offset=divmod(index,CHUNK_SIZE)
        if chunk not in self._mapped:
            self._mapped[chunk]=torch.load(self.folder/self._name(chunk),map_location='cpu',
                                           weights_only=True,mmap=True)
        return tuple(t[offset] for t in self._mapped[chunk])
