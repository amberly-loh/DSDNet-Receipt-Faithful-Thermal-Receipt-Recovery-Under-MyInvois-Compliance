"""Keep native OCR/model failures outside the HTTP server process."""
import atexit
import io
import os
import faulthandler
import multiprocessing as mp
from threading import Lock

_app = None


def _execute(payload):
    global _app
    from werkzeug.datastructures import FileStorage
    from .web_app import create_app
    if _app is None:
        _app = create_app(payload['config'],database_path=payload['database'],isolated_worker=True)
    upload = FileStorage(stream=io.BytesIO(payload['bytes']),filename=payload['filename'],content_type=payload['mimetype'])
    return _app.extensions['receipt_process_inline'](upload,payload['user'])


def _serve(connection,handler):
    if handler is _execute:
        faulthandler.enable()
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ.setdefault(name,'1')
        os.environ.setdefault('FLAGS_use_mkldnn','0')
        import cv2
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        import torch
        torch.set_num_threads(1)
    try:
        while True:
            payload = connection.recv()
            try:
                connection.send({'result':handler(payload)})
            except Exception as error:
                connection.send({'error':str(error)})
    except (EOFError,BrokenPipeError):
        pass
    finally:
        connection.close()


class IsolatedReceiptWorker:
    def __init__(self,handler=_execute,timeout=300):
        self.handler=handler
        self.timeout=timeout
        self.lock=Lock()
        self.process=None
        self.connection=None
        self.context=mp.get_context('spawn')
        atexit.register(self.close)

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection=None
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=3)
            self.process=None

    def run(self,payload):
        # Do not let concurrent scans occupy every HTTP thread while waiting
        # for the GPU. Sequential batches still reuse this worker normally.
        if not self.lock.acquire(blocking=False):
            raise ValueError('Another receipt is processing. Wait for it to finish, then retry this image.')
        try:
            if self.process is None or not self.process.is_alive():
                self.close()
                self.connection,child = self.context.Pipe()
                self.process=self.context.Process(target=_serve,args=(child,self.handler),daemon=True)
                self.process.start()
                child.close()
            try:
                self.connection.send(payload)
                if not self.connection.poll(self.timeout):
                    self.close()
                    raise ValueError('Receipt processing timed out. The app is still available; retry this image.')
                response=self.connection.recv()
            except (EOFError,BrokenPipeError,ConnectionResetError,OSError) as error:
                self.close()
                raise ValueError('The receipt processing worker stopped unexpectedly. The app is still available; retry this image.') from error
            if 'error' in response:
                raise ValueError(response['error'])
            return response['result']
        finally:
            self.lock.release()
