// Copy to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        // Show success feedback
        const feedback = document.createElement('div');
        feedback.className = 'copy-feedback';
        feedback.textContent = 'Copied to clipboard!';
        document.body.appendChild(feedback);
        
        setTimeout(() => {
            feedback.remove();
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy: ', err);
    });
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()" class="notification-close">&times;</button>
    `;
    
    document.querySelector('.notification-container').appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Toggle password visibility
function togglePasswordVisibility(inputId) {
    const input = document.getElementById(inputId);
    const toggleBtn = input.nextElementSibling;
    
    if (input.type === 'password') {
        input.type = 'text';
        toggleBtn.textContent = 'Hide';
    } else {
        input.type = 'password';
        toggleBtn.textContent = 'Show';
    }
}

// Initialize tooltips
function initTooltips() {
    const tooltips = document.querySelectorAll('[data-tooltip]');
    tooltips.forEach(element => {
        element.addEventListener('mouseenter', (e) => {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = e.target.dataset.tooltip;
            document.body.appendChild(tooltip);
            
            const rect = e.target.getBoundingClientRect();
            tooltip.style.top = `${rect.top - tooltip.offsetHeight - 10}px`;
            tooltip.style.left = `${rect.left + (rect.width - tooltip.offsetWidth) / 2}px`;
            
            e.target._tooltip = tooltip;
        });
        
        element.addEventListener('mouseleave', (e) => {
            if (e.target._tooltip) {
                e.target._tooltip.remove();
            }
        });
    });
}

// Form submission handler
async function submitForm(formId, endpoint) {
    const form = document.getElementById(formId);
    const formData = new FormData(form);
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification(result.message || 'Success!', 'success');
            if (result.redirect) {
                setTimeout(() => {
                    window.location.href = result.redirect;
                }, 1500);
            }
        } else {
            showNotification(result.message || 'An error occurred', 'error');
        }
    } catch (error) {
        showNotification('Network error. Please try again.', 'error');
    }
}

// Responsive menu toggle
function toggleMenu() {
    const menu = document.getElementById('mobile-menu');
    menu.classList.toggle('hidden');
}

// Initialize date picker
function initDatePicker() {
    flatpickr('.date-picker', {
        dateFormat: 'Y-m-d',
        minDate: 'today',
        disableMobile: true
    });
}

// Load more content (infinite scroll)
function initInfiniteScroll(containerId, loadMoreUrl) {
    const container = document.getElementById(containerId);
    let page = 1;
    let isLoading = false;
    
    window.addEventListener('scroll', debounce(() => {
        if (isLoading) return;
        
        const { scrollTop, scrollHeight, clientHeight } = document.documentElement;
        
        if (scrollTop + clientHeight >= scrollHeight - 100) {
            isLoading = true;
            page++;
            
            fetch(`${loadMoreUrl}?page=${page}`)
                .then(response => response.text())
                .then(html => {
                    container.insertAdjacentHTML('beforeend', html);
                    isLoading = false;
                })
                .catch(() => {
                    isLoading = false;
                });
        }
    }, 200));
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    initTooltips();
    
    // Initialize date pickers
    if (typeof flatpickr !== 'undefined') {
        initDatePicker();
    }
    
    // Add input event listeners for validation feedback
    document.querySelectorAll('input, select, textarea').forEach(input => {
        input.addEventListener('blur', function() {
            if (this.hasAttribute('required') && !this.value.trim()) {
                this.classList.add('invalid');
            } else if (this.value.trim()) {
                this.classList.remove('invalid');
                this.classList.add('valid');
            }
        });
    });
    
    // Handle form submissions
    document.querySelectorAll('form[data-ajax]').forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            const formId = this.id;
            const endpoint = this.dataset.endpoint;
            submitForm(formId, endpoint);
        });
    });
    
    // Initialize notifications container
    if (!document.querySelector('.notification-container')) {
        const container = document.createElement('div');
        container.className = 'notification-container';
        document.body.appendChild(container);
    }
});

// Export functions for module usage if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        validateForm,
        previewProfileImage,
        selectTimeSlot,
        selectDate,
        setRating,
        checkPasswordStrength,
        formatCurrency,
        debounce,
        copyToClipboard,
        showNotification,
        togglePasswordVisibility,
        initTooltips,
        submitForm,
        toggleMenu,
        initDatePicker,
        initInfiniteScroll
    };
}
