let jsonData = null;
let img = null;

function setError(message) {
    document.getElementById('error').textContent = message;
}

function clearError() {
    document.getElementById('error').textContent = '';
}

function updatePrivacyStatus() {
    const privacyLevel = document.getElementById('privacyLevel').value;
    document.getElementById('privacyStatus').textContent = `Текущий уровень приватности: ${privacyLevel}`;
}

document.getElementById('blurRadius').addEventListener('input', function(e) {
    document.getElementById('blurValue').textContent = e.target.value;
});

document.getElementById('privacyLevel').addEventListener('change', function(e) {
    if (jsonData && jsonData.employee) {
        jsonData.employee.privacy_level = e.target.value;
        console.log('Privacy level updated:', jsonData.employee.privacy_level);
        document.getElementById('downloadJsonButton').disabled = false;
        updatePrivacyStatus();
    } else {
        setError('JSON файл не загружен');
    }
});

document.getElementById('jsonFile').addEventListener('change', function(e) {
    clearError();
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            jsonData = JSON.parse(e.target.result);
            console.log('JSON loaded:', jsonData);
            if (jsonData.employee && jsonData.employee.privacy_level) {
                document.getElementById('privacyLevel').value = jsonData.employee.privacy_level;
                document.getElementById('downloadJsonButton').disabled = false;
                updatePrivacyStatus();
            } else {
                document.getElementById('privacyLevel').value = 'medium';
                jsonData.employee.privacy_level = 'medium';
                updatePrivacyStatus();
            }
        } catch (err) {
            setError('Ошибка парсинга JSON: ' + err.message);
        }
    };
    reader.onerror = function() {
        setError('Ошибка чтения JSON файла');
    };
    reader.readAsText(file);
});

document.getElementById('imageFile').addEventListener('change', function(e) {
    clearError();
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = function() {
            console.log('Image loaded:', img.naturalWidth, 'x', img.naturalHeight);
        };
        img.onerror = function() {
            setError('Ошибка загрузки изображения');
        };
        img.src = e.target.result;
    };
    reader.onerror = function() {
        setError('Ошибка чтения изображения');
    };
    reader.readAsDataURL(file);
});

async function testModule() {
    clearError();
    const rectX = parseInt(document.getElementById('rectX').value) || 50;
    const rectY = parseInt(document.getElementById('rectY').value) || 50;
    const fontSize = parseInt(document.getElementById('fontSize').value) || 30;
    const blurRadius = parseInt(document.getElementById('blurRadius').value) || 0;
    const fitMode = document.getElementById('fitMode').value;

    if (!jsonData) {
        setError('JSON файл не загружен');
        return;
    }
    if (!img) {
        setError('Изображение не загружено');
        return;
    }

    if (!img.complete || img.naturalWidth === 0 || img.naturalHeight === 0) {
        setError('Изображение не полностью загружено');
        return;
    }

    console.time('testModule');
    const outputCanvas = document.getElementById('outputCanvas');
    outputCanvas.width = 1920;
    outputCanvas.height = 1080;
    const resultCanvas = await processEmployeeData(jsonData, img, rectX, rectY, fontSize, blurRadius, fitMode);
    if (resultCanvas) {
        const ctx = outputCanvas.getContext('2d', { willReadFrequently: true });
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
        ctx.drawImage(resultCanvas, 0, 0);
    } else {
        setError('Ошибка обработки изображения');
    }
    console.timeEnd('testModule');
}

function downloadJson() {
    if (!jsonData) {
        setError('JSON файл не загружен');
        return;
    }
    const jsonStr = JSON.stringify(jsonData, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'employee_updated.json';
    a.click();
    URL.revokeObjectURL(url);
}

document.getElementById('testButton').addEventListener('click', testModule);
document.getElementById('downloadJsonButton').addEventListener('click', downloadJson);

updatePrivacyStatus();