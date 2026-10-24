"""Minimal backend-only provider probe; never print credentials."""
import config
import os
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError

if __name__ == '__main__':
    key=os.environ.get('GEMINI_API_KEY','')
    model=os.environ.get('GEMINI_MODEL','gemini-3.6-flash')
    print('Model:',model,'credential present:',bool(key))
    request=Request('https://generativelanguage.googleapis.com/v1beta/models/'+model+':generateContent',
        data=json.dumps({'contents':[{'parts':[{'text':'Reply OK'}]}]}).encode(),
        headers={'Content-Type':'application/json','x-goog-api-key':key})
    try:
        with urlopen(request,timeout=30) as response:
            print('Provider reachable:',response.status)
    except HTTPError as error:
        body=error.read().decode()
        print('HTTP',error.code,body.replace(key,'[REDACTED]')[:6000] if key else body[:6000])
