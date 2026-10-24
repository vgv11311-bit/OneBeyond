"""Private backup-key entry; does not replace the primary key or enable routing."""
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from provider_credentials import save_gemini_backup


def verify_and_save(key):
    if not key or any(c.isspace() for c in key):
        return False, 'Paste the complete API key without spaces.'
    try:
        request=Request('https://generativelanguage.googleapis.com/v1beta/models?pageSize=1',
                        headers={'x-goog-api-key':key,'User-Agent':'OneBeyond/1.0'})
        with urlopen(request,timeout=25) as response:
            data=json.load(response)
        if not isinstance(data,dict) or not data.get('models'):
            return False, 'Google returned no models. The key has not been saved.'
        save_gemini_backup(key)
        return True, ('Backup key verified and encrypted locally. Primary key unchanged.\n'
                      'Generation quota and automatic fallback still need verification.')
    except HTTPError as error:
        code=error.code
        error.close()
        if code in (401,403):
            return False, 'Google denied access. Check the key and Gemini API permissions.'
        if code==429:
            return False, 'Google reported a usage limit. Nothing saved; try again later.'
        return False, f'Google verification failed (HTTP {code}). Nothing saved.'
    except Exception:
        return False, 'Could not verify or save securely. Check your connection and retry.'


def main():
    root=tk.Tk()
    root.title('One Beyond — private Gemini backup setup')
    root.geometry('660x340')
    frame=ttk.Frame(root,padding=22)
    frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='Add your second Gemini key',font=('Segoe UI',16,'bold')).pack(anchor='w')
    ttk.Label(frame,text=('Your existing key remains priority 1. This key is saved separately as backup.\n'
                          'It is sent only to Google for verification and stored encrypted for your\n'
                          'Windows account—not in chat, source code, or the web app frontend.'),
              justify='left').pack(anchor='w',pady=14)
    ttk.Label(frame,text='Backup Gemini API key (new project)').pack(anchor='w')
    entry=ttk.Entry(frame,show='*',width=80)
    entry.pack(fill='x',pady=6)
    messages=queue.Queue()
    status=tk.StringVar(value='Ready for private entry. Do not paste your key into chat.')
    def submit():
        key=entry.get().strip()
        button.config(state='disabled')
        status.set('Verifying access with Google…')
        threading.Thread(target=lambda:messages.put(verify_and_save(key)),daemon=True).start()
    button=ttk.Button(frame,text='Verify and save backup securely',command=submit)
    button.pack(anchor='w',pady=12)
    ttk.Label(frame,textvariable=status,wraplength=610).pack(anchor='w')
    def poll():
        try:
            success,message=messages.get_nowait()
            status.set(message)
            if success: entry.delete(0,'end')
            button.config(state='normal')
        except queue.Empty: pass
        root.after(150,poll)
    poll()
    root.mainloop()

if __name__=='__main__': main()
