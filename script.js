function checkAge() {
    var ageInput = document.getElementById('ageInput').value;
    var contentDiv = document.getElementById('content');
    if (ageInput >= 18) {
        contentDiv.classList.add('show');
    } else {
        contentDiv.classList.remove('show');
    }
    contentDiv.classList.add('blur');
}

