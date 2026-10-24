"""Private local setup UI. No key in chat, command arguments, logs or frontend."""
import json
import queue
import re
import threading
import tkinter as tk
from tkinter import ttk
import webbrowser
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from provider_credentials import DEFAULT_GROQ_MODEL, save_groq_credentials
from groq_http import groq_headers, setup_error_message

def main():
    root=tk.Tk()
    root.title('One Beyond — secure Groq setup (updated)')
    root.geometry('630x410')
    frame=ttk.Frame(root,padding=22)
    frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='Connect Groq privately',font=('Segoe UI',16,'bold')).pack(anchor='w')
    ttk.Label(frame,text='First revoke the key posted in chat, then create a replacement.\nPaste the NEW key below. It stays on this computer, encrypted for your\nWindows account. It is sent only to Groq to verify access.',justify='left').pack(anchor='w',pady=12)
    ttk.Button(frame,text='Open Groq API Keys',command=lambda:webbrowser.open('https://console.groq.com/keys')).pack(anchor='w')
    ttk.Label(frame,text='New API key').pack(anchor='w',pady=(12,3))
    key_entry=ttk.Entry(frame,show='*',width=80)
    key_entry.pack(fill='x')
    ttk.Label(frame,text='Vision model (editable)').pack(anchor='w',pady=(12,3))
    model_entry=ttk.Entry(frame,width=80)
    model_entry.insert(0,DEFAULT_GROQ_MODEL)
    model_entry.pack(fill='x')
    status=tk.StringVar(value='No credential has been saved by this window yet.')
    messages=queue.Queue()

    def connect():
        key=key_entry.get().strip()
        model=model_entry.get().strip()
        if not key.startswith('gsk_') or not re.fullmatch(r'[A-Za-z0-9._/-]+',model):
            status.set('Enter a valid new Groq key and model ID.'); return
        button.config(state='disabled')
        status.set('Checking credential and model access with Groq…')
        def work():
            try:
                request=Request('https://api.groq.com/openai/v1/models',headers=groq_headers(key))
                with urlopen(request,timeout=20) as response: models=json.load(response)
                if model not in {item.get('id') for item in models.get('data',[])}:
                    messages.put((False,'That model is not listed for this account. Check the model ID.')); return
                save_groq_credentials(key,model)
                messages.put((True,'Connected and encrypted. In Create 3D Space, select Groq Vision.\nModel access verified; room quality still needs a test upload.'))
            except HTTPError as error:
                messages.put((False,setup_error_message(error)))
            except Exception:
                messages.put((False,'Could not connect or save securely. Check your connection and try again.'))
        threading.Thread(target=work,daemon=True).start()

    button=ttk.Button(frame,text='Verify and save securely',command=connect)
    button.pack(anchor='w',pady=14)
    ttk.Label(frame,textvariable=status,wraplength=575).pack(anchor='w')
    def poll():
        try:
            success,message=messages.get_nowait()
            if success: key_entry.delete(0,'end')
            status.set(message)
            button.config(state='normal')
        except queue.Empty: pass
        root.after(150,poll)
    poll()
    root.mainloop()

if __name__=='__main__': main()
