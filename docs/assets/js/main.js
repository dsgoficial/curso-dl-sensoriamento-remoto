document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling para links internos
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Highlight do TOC baseado na posição do scroll
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const id = entry.target.getAttribute('id');
            const tocLink = document.querySelector(`.toc a[href="#${id}"]`);
            
            if (entry.isIntersecting) {
                document.querySelectorAll('.toc a').forEach(link => {
                    link.classList.remove('active');
                });
                if (tocLink) {
                    tocLink.classList.add('active');
                }
            }
        });
    }, {
        rootMargin: '-20% 0% -35% 0%'
    });
    
    // Observar todos os headings
    document.querySelectorAll('h1[id], h2[id], h3[id], h4[id]').forEach(heading => {
        observer.observe(heading);
    });
    
    // Copy button para código
    document.querySelectorAll('.highlight pre').forEach(pre => {
        const button = document.createElement('button');
        button.textContent = 'Copiar';
        button.className = 'copy-btn';
        button.addEventListener('click', () => {
            navigator.clipboard.writeText(pre.textContent).then(() => {
                button.textContent = 'Copiado!';
                setTimeout(() => {
                    button.textContent = 'Copiar';
                }, 2000);
            });
        });
        
        const wrapper = document.createElement('div');
        wrapper.className = 'code-wrapper';
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(button);
        wrapper.appendChild(pre);
    });
});