# Source package validation

Checked locally on 1 November 2026.

- TypeScript and Vite production build: passed.
- Frontend unit tests: 45 passed.
- Node HTTP proxy test: 1 passed.
- Flask import and local health, root and saved-space routes: passed.
- Backend unit tests: 76 passed.
- Engine discovery exposes only COLMAP + ReSplat; renderer module route returns HTTP 200.
- No bundled sample is advertised as a generated room; sample-generation route returns a descriptive HTTP 404.

Build warnings remain for large JavaScript chunks. These checks do not validate
CUDA installation, cloud credentials, a newly trained room, or restored scene data.
No GPU session or website deployment was performed for packaging.

Run locally:

```powershell
python -m unittest discover -s backend -p "test_*.py"
cd frontend
npm run build
npm test
```

Some application-snapshot backend/experiment tests require external models or
private fixtures. The simplified package uses synthetic input in its video test.
