import { createSiteServer } from './server.mjs';

// Hostinger imports its entry module; it does not invoke it as a CLI script.
createSiteServer().listen(Number(process.env.PORT || 3000), '0.0.0.0');
