"""Unauthenticated transport diagnostic. Does not read any credentials."""
from urllib.request import Request, urlopen
from urllib.error import HTTPError

if __name__=='__main__':
    for headers in ({},{'User-Agent':'OneBeyond/1.0','Accept':'application/json'}):
        try:
            with urlopen(Request('https://api.groq.com/openai/v1/models',headers=headers),timeout=15) as response:
                print('Explicit app identification:',bool(headers),'status:',response.status)
        except HTTPError as error:
            print('Explicit app identification:',bool(headers),'status:',error.code,'response:',error.read(800).decode(errors='replace'))
