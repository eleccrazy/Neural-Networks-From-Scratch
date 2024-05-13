document.addEventListener('DOMContentLoaded', function() {
  const inputMethodCheckbox = document.getElementById('input-method');
  const manualInputContainer = document.getElementById('manual-input-container');
  const fileInputContainer = document.getElementById('file-input-container');

  // Toggle initial visibility based on the checkbox state
  if (inputMethodCheckbox.checked) {
    manualInputContainer.style.display = 'none';
    fileInputContainer.style.display = 'block';
  } else {
    manualInputContainer.style.display = 'block';
    fileInputContainer.style.display = 'none';
  }

  inputMethodCheckbox.addEventListener('change', function() {
    if (this.checked) {
      manualInputContainer.style.display = 'none';
      fileInputContainer.style.display = 'block';
    } else {
      manualInputContainer.style.display = 'block';
      fileInputContainer.style.display = 'none';
    }
  });
});
