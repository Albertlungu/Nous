
// View switching functionality
document.addEventListener('DOMContentLoaded', () => {
    const navButtons = document.querySelectorAll('.nav-btn');
    const views = document.querySelectorAll('.view');

    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const viewName = button.getAttribute('data-view');

            navButtons.forEach(btn => btn.classList.remove('active'));
            views.forEach(view => view.classList.remove('active'));

            button.classList.add('active');
            document.getElementById(`${viewName}-view`).classList.add('active');
        });
    });
});

// Toggle generation panel
document.addEventListener('DOMContentLoaded', () => {
    const genSettingsBtn = document.getElementById('gen-settings');
    const generationPanel = document.getElementById('generation-panel');

    if (genSettingsBtn && generationPanel) {
        genSettingsBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            generationPanel.classList.toggle('show');
            genSettingsBtn.classList.toggle('active');
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!generationPanel.contains(e.target) && e.target !== genSettingsBtn) {
                generationPanel.classList.remove('show');
                genSettingsBtn.classList.remove('active');
            }
        });
    }

    // Expandable sidebar
    const sidebar = document.getElementById('sidebar');

    if (sidebar) {
        sidebar.addEventListener('mouseenter', () => {
            sidebar.classList.add('expanded');
        });

        sidebar.addEventListener('mouseleave', () => {
            sidebar.classList.remove('expanded');
        });
    }

    // Theme toggle
    const themeToggle = document.getElementById('theme-toggle');

    // Load saved theme preference
    const savedTheme = localStorage.getItem('theme') || 'dark';
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
        themeToggle.textContent = '☾';
    }

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            document.body.classList.toggle('light-theme');
            const isLight = document.body.classList.contains('light-theme');

            // Update icon
            themeToggle.textContent = isLight ? '☾' : '☀';

            // Save preference
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
        });
    }
});
