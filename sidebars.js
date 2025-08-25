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
            'modulos/modulo1/numpy',
            'modulos/modulo1/matplotlib_opencv',
            'modulos/modulo1/processamento-imagens',
          ],
        },
        {
          type: 'category',
          label: 'Módulo 2: Redes Neurais (4h)',
          items: [
            'modulos/modulo2/visao_geral',
            'modulos/modulo2/pytorch_vs_numpy',
            'modulos/modulo2/calculo_dl',
            'modulos/modulo2/perceptron',
          ],
        },
        {
          type: 'category',
          label: 'Módulo 3: Treinamento de Redes Neurais (4h)',
          items: [
            'modulos/modulo3/visao_geral',
            'modulos/modulo3/training_loop',
            'modulos/modulo3/dataloader',
            'modulos/modulo3/losses',
            'modulos/modulo3/learning_rate_schedulers',
            'modulos/modulo3/checkpointing',
            'modulos/modulo3/avaliacao_treinamento',
            'modulos/modulo3/regularizers',
            'modulos/modulo3/treinamento_completo',
          ],
        },
        {
          type: 'category',
          label: 'Módulo 4: CNNs (4h)',
          items: [
            'modulos/modulo4/visao_geral',
            'modulos/modulo4/cnn_limitacoes_mlp',
            'modulos/modulo4/cnn_architecture',
            'modulos/modulo4/convolution_details',
            'modulos/modulo4/lenet_mnist',
            'modulos/modulo4/cnn_vs_mlp',
            'modulos/modulo4/cnn_visualization',
            'modulos/modulo4/cnn_embeddings',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Segunda Semana - Presencial (40h)',
      items: [
        {
          type: 'category',
          label: 'Módulo 5: Arquiteturas CNN Avançadas (8h)',
          items: [
            'modulos/modulo5/visao_geral',
            'modulos/modulo5/alexnet',
            'modulos/modulo5/inception',
            'modulos/modulo5/vgg_family',
            'modulos/modulo5/resnet',
            'modulos/modulo5/transfer_learning',
          ],
        },
      ],
    },
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