export type SiteRoute = 'home' | 'studio' | 'sample' | 'recent' | 'about' | 'journal' | 'contact';

const routePaths: Record<SiteRoute, string> = {
  home: '/',
  studio: '/studio',
  sample: '/sample-space',
  recent: '/spaces',
  about: '/about',
  journal: '/journal',
  contact: '/contact',
};

export const sitePath = (route: SiteRoute) => routePaths[route];

export const currentSiteRoute = (): SiteRoute => {
  const path = window.location.pathname.replace(/\/+$/, '') || '/';
  return (Object.keys(routePaths) as SiteRoute[]).find(route => routePaths[route] === path) || 'home';
};

export const navigateSite = (route: SiteRoute) => {
  const path = sitePath(route);
  if (window.location.pathname !== path) window.history.pushState({}, '', path);
  window.dispatchEvent(new PopStateEvent('popstate'));
};
