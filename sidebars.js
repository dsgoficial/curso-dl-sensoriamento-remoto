const sidebars = {
  // Sidebar principal do curso
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Primeira Semana - EAD (16h)',
      items: [
        {
          type: 'category',
          label: 'Módulo 1: Fundamentos (4h)',
          items: [
            'modulos/modulo1/index',
            'modulos/modulo1/introducao',
            'modulos/modulo1/setup',
            'modulos/modulo1/matematica',
            'modulos/modulo1/processamento-imagens',
          ],
        },
        {
          type: 'category',
          label: 'Módulo 2: Redes Neurais (4h)',
          items: [
            'modulos/modulo2/visao_geral',
            // 'modulos/modulo2/treinamento-pytorch',
          ],
        },
        // {
        //   type: 'category',
        //   label: 'Módulo 3: Treinamento (4h)',
        //   items: [
        //     'modulos/modulo3/training-loop',
        //     'modulos/modulo3/fenomenos-treinamento',
        //   ],
        // },
        // {
        //   type: 'category',
        //   label: 'Módulo 4: CNNs (4h)',
        //   items: [
        //     'modulos/modulo4/convolucao-classica-cnns',
        //     'modulos/modulo4/lenet-mnist',
        //   ],
        // },
      ],
    },
    // {
    //   type: 'category',
    //   label: 'Segunda Semana - Presencial (40h)',
    //   items: [
    //     'presencial/dia1',
    //     'presencial/dia2',
    //     'presencial/dia3',
    //     'presencial/dia4',
    //     'presencial/dia5',
    //   ],
    // },
    // {
    //   type: 'category',
    //   label: 'Recursos',
    //   items: [
    //     'recursos/datasets',
    //     'recursos/bibliografia',
    //     'recursos/faq',
    //   ],
    // },
  ],
};

module.exports = sidebars;