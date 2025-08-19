const config = {
  title: 'Deep Learning Aplicado ao Sensoriamento Remoto',
  tagline: 'Curso avançado para mestrado e doutorado - Cap Philipe Borba',
  
  url: 'https://dsgoficial.github.io',
  baseUrl: '/curso-dl-sensoriamento-remoto/',
  
  organizationName: 'dsgoficial',
  projectName: 'curso-dl-sensoriamento-remoto',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,
  
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  
  i18n: {
    defaultLocale: 'pt-BR',
    locales: ['pt-BR'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // SEM plugins matemáticos por enquanto
          editUrl: 'https://github.com/dsgoficial/curso-dl-sensoriamento-remoto/tree/main/',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'DL + Sensoriamento Remoto',
      items: [
        {
          type: 'doc',
          docId: 'intro',
          position: 'left',
          label: 'Curso',
        },
        {
          href: 'https://github.com/dsgoficial/curso-dl-sensoriamento-remoto',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    
    footer: {
      style: 'dark',
      copyright: `Copyright © ${new Date().getFullYear()} Cap Philipe Borba. Construído com Docusaurus.`,
    },
    
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
  },
};

module.exports = config;