import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from provider_credentials import save_gemini_backup, gemini_backup_key
from setup_gemini_backup import verify_and_save

class BackupSetupTests(unittest.TestCase):
    def test_verification_uses_header_and_saves_only_after_success(self):
        with patch('setup_gemini_backup.urlopen',return_value=io.BytesIO(b'{"models":[{"name":"test"}]}')) as network, patch('setup_gemini_backup.save_gemini_backup') as save:
            self.assertTrue(verify_and_save('synthetic-secret')[0])
            request=network.call_args.args[0]
            self.assertNotIn('synthetic-secret',request.full_url)
            self.assertEqual(request.get_header('X-goog-api-key'),'synthetic-secret')
            save.assert_called_once_with('synthetic-secret')

    def test_failure_does_not_save_or_expose_secret(self):
        for code in (401,403,429,503):
            with patch('setup_gemini_backup.urlopen',side_effect=HTTPError('url',code,'synthetic-secret',{},io.BytesIO(b'synthetic-secret'))), patch('setup_gemini_backup.save_gemini_backup') as save:
                success,message=verify_and_save('synthetic-secret')
                self.assertFalse(success)
                self.assertNotIn('synthetic-secret',message)
                save.assert_not_called()

    @unittest.skipUnless(os.name=='nt','Windows encrypted storage')
    def test_backup_encryption_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder, patch('provider_credentials.gemini_backup_path',return_value=Path(folder)/'backup.dpapi'):
            save_gemini_backup('synthetic-secret')
            self.assertEqual(gemini_backup_key(),'synthetic-secret')
            self.assertNotIn(b'synthetic-secret',(Path(folder)/'backup.dpapi').read_bytes())

if __name__=='__main__': unittest.main()
