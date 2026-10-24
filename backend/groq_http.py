"""Shared identification and safe diagnostics for the Groq API client."""
import json

def groq_headers(key):
    return {'Authorization':'Bearer '+key,'User-Agent':'OneBeyond/1.0',
            'Accept':'application/json','Content-Type':'application/json'}

def setup_error_message(error):
    # Never display raw provider bodies: they could echo credentials or HTML.
    try: body=error.read(8192).decode(errors='replace')
    except Exception: body=''
    if error.code==403 and '1010' in body:
        return 'Connection blocked before authentication (HTTP 403/1010). Use the updated setup window; check network restrictions if it persists.'
    if error.code==401:
        return 'Groq did not accept this key (HTTP 401). Check that you copied the complete new key.'
    if error.code==403:
        return 'Groq denied access (HTTP 403). Check organization/project permissions in Groq Console; this does not prove the key is invalid.'
    if error.code==429:
        return 'Groq rate limit reached (HTTP 429). Wait before trying setup again.'
    return f'Groq could not complete setup (HTTP {error.code}). Try later or check your account.'
