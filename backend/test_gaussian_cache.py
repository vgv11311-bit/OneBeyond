import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reconstruction.gaussian_cache import training_key,publish,restore,verified_t4_key
from reconstruction import gaussian_cache


class GaussianCacheTests(unittest.TestCase):
    def test_video_selection_evidence_is_hash_verified_and_restored(self):
        path=self.source/'poses/video_selection.json'
        path.write_text(json.dumps(dict(candidate_count=2,selected_count=1,frames=[dict(status='selected'),dict(status='duplicate')])))
        publish(self.cache,self.key,self.source)
        self.assertTrue(restore(self.cache,self.key,self.target))
        self.assertEqual((self.target/'poses/video_selection.json').read_bytes(),path.read_bytes())
        (self.cache/self.key/'poses/video_selection.json').write_text('{}')
        self.assertFalse(restore(self.cache,self.key,self.root/'other'))
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.source=self.root/'source';self.source.mkdir()
        self.photo=self.root/'photo.jpg';self.photo.write_bytes(b'input photo')
        self.cache=self.root/'cache';self.target=self.root/'target';self.target.mkdir()
        for name in ['scene.ply','config.json','poses/inputs.json','poses/images/0.jpg',
                     'poses/brush_dataset/images/0.jpg','poses/brush_dataset/sparse/images.txt',
                     'poses/model_text_0/images.txt','poses/model_text_0/cameras.txt','poses/model_text_0/points3D.txt']:
            path=self.source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
        (self.source/'manifest.json').write_text(json.dumps(dict(status='ready',vertices=100)))
        self.key=training_key([self.photo],{'brush':'fixture-tool'})

    def test_identical_bytes_ignore_upload_filename_but_not_order_or_content(self):
        other=self.root/'renamed.jpg';other.write_bytes(self.photo.read_bytes())
        self.assertEqual(self.key,training_key([other],{'brush':'fixture-tool'}))
        other.write_bytes(b'new photo')
        self.assertNotEqual(self.key,training_key([other],{'brush':'fixture-tool'}))
        self.assertNotEqual(training_key([other,self.photo],{}),training_key([self.photo,other],{}))

    def test_tools_and_implementation_invalidate(self):
        self.assertNotEqual(self.key,training_key([self.photo],{'brush':'different-tool'}))
        with patch('reconstruction.gaussian_cache.TRAINING_FILES',[]):
            self.assertNotEqual(self.key,training_key([self.photo],{'brush':'fixture-tool'}))

    def test_verified_t4_provenance_does_not_impersonate_local_training(self):
        key=verified_t4_key([self.photo])
        self.assertNotEqual(key,training_key([self.photo],{'brush':'fixture-tool'},'standalone'))
        renamed=self.root/'same.jpg';renamed.write_bytes(self.photo.read_bytes())
        self.assertEqual(key,verified_t4_key([renamed]))

    def photo_set_fixture(self):
        spaces=self.root/'spaces';spaces.mkdir();uploads=self.root/'uploads';uploads.mkdir()
        files=[]
        for i in range(3):
            photo=uploads/f'{i}.jpg';photo.write_bytes(f'original photo {i}'.encode());files.append(photo)
        room=spaces/'old-room';room.mkdir()
        (room/'metadata.json').write_text(json.dumps(dict(status='completed',input_type='photos',file_paths=list(map(str,files)))))
        publish(self.cache,verified_t4_key(files),self.source)
        return spaces,files

    def test_verified_photo_set_reuses_reordered_identical_bytes_and_keeps_camera_mapping(self):
        spaces,files=self.photo_set_fixture()
        self.assertTrue(gaussian_cache.restore_verified_photo_set(self.cache,files[::-1],self.target,spaces))
        self.assertEqual((self.target/'poses/inputs.json').read_bytes(),b'fixture')
        proof=json.loads((self.target/'reuse-provenance.json').read_text())
        self.assertTrue(proof['input_order_changed'])

    def test_verified_photo_set_rejects_changed_duplicate_or_untrusted_inputs(self):
        spaces,files=self.photo_set_fixture()
        self.assertFalse(gaussian_cache.restore_verified_photo_set(self.cache,[files[0],files[0],files[2]],self.target,spaces))
        files[1].write_bytes(b'changed pixels')
        self.assertFalse(gaussian_cache.restore_verified_photo_set(self.cache,files[::-1],self.target,spaces))
        self.assertEqual(list(self.target.iterdir()),[])

    def test_verified_photo_set_rejects_corrupt_training(self):
        spaces,files=self.photo_set_fixture()
        (self.cache/verified_t4_key(files)/'scene.ply').write_bytes(b'corrupt')
        self.assertFalse(gaussian_cache.restore_verified_photo_set(self.cache,files[::-1],self.target,spaces))
        self.assertEqual(list(self.target.iterdir()),[])

    def test_verified_photo_set_rejects_original_metadata_paths_outside_uploads(self):
        spaces,files=self.photo_set_fixture();outside=self.root/'outside';outside.mkdir()
        paths=[]
        for i,file in enumerate(files):
            copy=outside/f'{i}.jpg';copy.write_bytes(file.read_bytes());paths.append(str(copy))
        (spaces/'old-room/metadata.json').write_text(json.dumps(dict(status='completed',input_type='photos',file_paths=paths)))
        self.assertFalse(gaussian_cache.restore_verified_photo_set(self.cache,files[::-1],self.target,spaces))
        self.assertEqual(list(self.target.iterdir()),[])

    def test_complete_hit_is_byte_identical_and_does_not_share_writable_files(self):
        self.assertTrue(publish(self.cache,self.key,self.source))
        self.assertTrue(restore(self.cache,self.key,self.target))
        self.assertEqual((self.target/'scene.ply').read_bytes(),(self.source/'scene.ply').read_bytes())
        (self.target/'scene.ply').write_bytes(b'cleanup current result')
        self.assertEqual((self.cache/self.key/'scene.ply').read_bytes(),b'fixture')

    def test_corrupt_cache_is_miss_before_any_output_is_copied(self):
        publish(self.cache,self.key,self.source)
        (self.cache/self.key/'scene.ply').write_bytes(b'corrupt')
        self.assertFalse(restore(self.cache,self.key,self.target))
        self.assertEqual(list(self.target.iterdir()),[])

    def test_incomplete_training_is_never_cached(self):
        (self.source/'manifest.json').write_text('{"status":"running"}')
        with self.assertRaises(ValueError):publish(self.cache,self.key,self.source)
        self.assertFalse(restore(self.cache,self.key,self.target))

    def test_traversal_in_manifest_rejected(self):
        publish(self.cache,self.key,self.source)
        metadata=self.cache/self.key/'cache.json';payload=json.loads(metadata.read_text())
        payload['files']['../outside']='hash';metadata.write_text(json.dumps(payload))
        self.assertFalse(restore(self.cache,self.key,self.target))


if __name__=='__main__':unittest.main()
