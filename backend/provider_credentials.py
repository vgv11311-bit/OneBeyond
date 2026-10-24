"""Backend-only Windows user-scoped DPAPI credentials; env overrides supported."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path

DEFAULT_GROQ_MODEL='qwen/qwen3.8-27b'

def gemini_backup_path():
    base=os.environ.get('LOCALAPPDATA')
    if not base: raise RuntimeError('Windows local storage is unavailable.')
    return Path(base)/'OneBeyond'/'gemini-backup.dpapi'

def save_gemini_backup(key):
    path=gemini_backup_path()
    encrypted=crypt(json.dumps({'key':key}).encode())
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_bytes(encrypted)
    temporary.replace(path)

def gemini_backup_key():
    path=gemini_backup_path()
    if not path.exists(): return ''
    return json.loads(crypt(path.read_bytes(),decrypt=True))['key']

def credential_path():
    base=os.environ.get('LOCALAPPDATA')
    if not base: raise RuntimeError('Use GROQ_API_KEY and GROQ_MODEL on non-Windows hosts.')
    return Path(base)/'OneBeyond'/'groq.dpapi'

class Blob(ctypes.Structure):
    _fields_=[('size',wintypes.DWORD),('data',ctypes.POINTER(ctypes.c_ubyte))]

def crypt(data, decrypt=False):
    if os.name!='nt': raise RuntimeError('Encrypted local credential storage requires Windows.')
    buffer=ctypes.create_string_buffer(data)
    source=Blob(len(data),ctypes.cast(buffer,ctypes.POINTER(ctypes.c_ubyte)))
    output=Blob()
    api=ctypes.WinDLL('crypt32',use_last_error=True)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    function=api.CryptUnprotectData if decrypt else api.CryptProtectData
    function.argtypes=[ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
    function.restype=wintypes.BOOL
    kernel.LocalFree.argtypes=[ctypes.c_void_p]
    kernel.LocalFree.restype=ctypes.c_void_p
    if not function(ctypes.byref(source),None,None,None,None,1,ctypes.byref(output)):
        raise RuntimeError('Windows could not protect/unprotect the provider credential.')
    try: return ctypes.string_at(output.data,output.size)
    finally: kernel.LocalFree(output.data)

def save_groq_credentials(key, model):
    path=credential_path()
    path.parent.mkdir(parents=True,exist_ok=True)
    encrypted=crypt(json.dumps({'key':key,'model':model}).encode())
    temporary=path.with_suffix('.tmp')
    temporary.write_bytes(encrypted)
    temporary.replace(path)

def groq_credentials():
    key=os.environ.get('GROQ_API_KEY','').strip()
    if key: return key,os.environ.get('GROQ_MODEL',DEFAULT_GROQ_MODEL).strip()
    try:
        path=credential_path()
        if not path.exists(): return '',os.environ.get('GROQ_MODEL',DEFAULT_GROQ_MODEL)
        saved=json.loads(crypt(path.read_bytes(),decrypt=True))
        return saved['key'],os.environ.get('GROQ_MODEL',saved.get('model',DEFAULT_GROQ_MODEL))
    except (OSError,ValueError,KeyError,RuntimeError):
        raise RuntimeError('Groq credentials are unavailable. Run the local Groq setup again.') from None
