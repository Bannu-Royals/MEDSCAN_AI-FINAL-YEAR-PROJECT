document.addEventListener('DOMContentLoaded', function () {
    const tabs = document.querySelectorAll('.service-tab');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', function () {
            const targetTab = this.getAttribute('data-tab');
            console.log(`Clicked tab: ${targetTab}`); // Debugging log

            // Remove active class from all tabs
            tabs.forEach(t => {
                t.classList.remove('active');
                t.classList.remove('text-primary');
                t.classList.add('text-text-light');
                t.classList.remove('border-primary');
                t.classList.add('border-gray-300');
            });

            // Add active class to the clicked tab
            this.classList.add('active');
            this.classList.remove('text-text-light');
            this.classList.add('text-primary');
            this.classList.remove('border-gray-300');
            this.classList.add('border-primary');

            // Hide all tab contents
            tabContents.forEach(content => {
                content.classList.remove('active');
                console.log(`Hiding content: ${content.id}`); // Debugging log
            });

            // Show the target tab content
            const targetContent = document.getElementById(`${targetTab}-content`);
            if (targetContent) {
                targetContent.classList.add('active');
                console.log(`Showing content: ${targetContent.id}`); // Debugging log
            } else {
                console.error(`Target content not found: ${targetTab}-content`);
            }
        });
    });
});
document.addEventListener('DOMContentLoaded', function () {
setTimeout(() => {
const counters = document.querySelectorAll('.counter');

counters.forEach(counter => {
    const target = +counter.getAttribute('data-target');
    const duration = 4000;
    const increment = target / (duration / 16);

    let current = 0;

    const updateCounter = () => {
        current += increment;
        if (current < target) {
            counter.textContent = Math.ceil(current);
            requestAnimationFrame(updateCounter);
        } else {
            counter.textContent = target;
        }
    };

    updateCounter();
});
}, 500); // Start after 500ms delay
});

document.addEventListener('DOMContentLoaded', function () {
const mobileMenuButton = document.getElementById('mobileMenuButton');
const mobileMenu = document.getElementById('mobileMenu');

if (mobileMenuButton && mobileMenu) {
mobileMenuButton.addEventListener('click', function () {
    mobileMenu.classList.toggle('hidden'); // Toggle the 'hidden' class
});
}
});