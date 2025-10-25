async function convertImagePalette(image, hex_colors, black_threshold = 0.37, white_threshold = 0.25, fitMode = 'fit') {
    console.time('convertImagePalette');
    console.log('Starting convertImagePalette for image:', image.src);

    // Проверка загрузки изображения
    if (!image.complete || image.naturalWidth === 0 || image.naturalHeight === 0) {
        console.error('Изображение не загружено или повреждено:', image.src);
        console.timeEnd('convertImagePalette');
        return null;
    }
    console.log('Изображение загружено:', image.src);
    console.log('Image dimensions:', image.naturalWidth, 'x', image.naturalHeight);

    // Создаем промежуточный canvas для нормализации
    const intermediateCanvas = document.createElement('canvas');
    intermediateCanvas.width = 1920;
    intermediateCanvas.height = 1080;
    const intermediateCtx = intermediateCanvas.getContext('2d', { willReadFrequently: true });
    intermediateCtx.imageSmoothingEnabled = false;
    intermediateCtx.clearRect(0, 0, 1920, 1080);

    // Обрабатываем изображение в зависимости от режима
    try {
        console.log('Drawing image to intermediate canvas, mode:', fitMode);
        const imgWidth = image.naturalWidth;
        const imgHeight = image.naturalHeight;
        const targetWidth = 1920;
        const targetHeight = 1080;

        if (fitMode === 'stretch') {
            // Растягиваем изображение до 1920x1080
            intermediateCtx.drawImage(image, 0, 0, targetWidth, targetHeight);
        } else if (fitMode === 'crop') {
            // Обрезаем, центрируя изображение
            const ratio = Math.min(targetWidth / imgWidth, targetHeight / imgHeight);
            const newWidth = imgWidth * ratio;
            const newHeight = imgHeight * ratio;
            const offsetX = (targetWidth - newWidth) / 2;
            const offsetY = (targetHeight - newHeight) / 2;
            intermediateCtx.drawImage(
                image,
                0, 0, imgWidth, imgHeight,
                offsetX, offsetY, newWidth, newHeight
            );
        } else {
            // Режим 'fit' (вписать, сохраняя пропорции)
            const ratio = Math.min(targetWidth / imgWidth, targetHeight / imgHeight);
            const newWidth = imgWidth * ratio;
            const newHeight = imgHeight * ratio;
            const offsetX = (targetWidth - newWidth) / 2;
            const offsetY = (targetHeight - newHeight) / 2;
            intermediateCtx.drawImage(
                image,
                0, 0, imgWidth, imgHeight,
                offsetX, offsetY, newWidth, newHeight
            );
            // Заполняем фон черным, если изображение не покрывает весь canvas
            intermediateCtx.fillStyle = 'black';
            intermediateCtx.fillRect(0, 0, targetWidth, targetHeight);
            intermediateCtx.drawImage(
                image,
                0, 0, imgWidth, imgHeight,
                offsetX, offsetY, newWidth, newHeight
            );
        }
    } catch (e) {
        console.error('Ошибка отрисовки изображения:', e.message);
        console.timeEnd('convertImagePalette');
        return null;
    }

    // Создаем основной canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 1920;
    tempCanvas.height = 1080;
    const ctx = tempCanvas.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, 1920, 1080);

    // Копируем данные
    try {
        console.log('Copying image to main canvas');
        ctx.drawImage(intermediateCanvas, 0, 0, 1920, 1080);
    } catch (e) {
        console.error('Ошибка копирования изображения:', e.message);
        console.timeEnd('convertImagePalette');
        return null;
    }

    // Получаем пиксельные данные
    let imageData;
    try {
        console.log('Getting image data');
        imageData = ctx.getImageData(0, 0, 1920, 1080);
    } catch (e) {
        console.error('Ошибка получения пиксельных данных:', e.message);
        console.timeEnd('convertImagePalette');
        return null;
    }
    const data = imageData.data;
    const newData = new Uint8ClampedArray(data.length);

    // Логируем первые пиксели
    console.log('Первые пиксели (до обработки):',
        `R: ${data[0]}, G: ${data[1]}, B: ${data[2]}, A: ${data[3]}`);

    // Копируем пиксельные данные
    for (let i = 0; i < data.length; i += 4) {
        newData[i] = data[i];
        newData[i + 1] = data[i + 1];
        newData[i + 2] = data[i + 2];
        newData[i + 3] = 255;
    }

    // Преобразуем hex-цвета в RGB
    function hexToRgb(hex_str) {
        try {
            if (hex_str.startsWith('#')) {
                hex_str = hex_str.slice(1);
            }
            if (hex_str.length !== 6) {
                throw new Error(`Неверный hex-цвет: ${hex_str}`);
            }
            return [
                parseInt(hex_str.substr(0, 2), 16) / 255,
                parseInt(hex_str.substr(2, 2), 16) / 255,
                parseInt(hex_str.substr(4, 2), 16) / 255
            ];
        } catch (e) {
            console.error(e.message);
            return [0, 0, 0];
        }
    }
    const rgb_colors = hex_colors.map(h => hexToRgb(h));
    console.log('RGB colors:', rgb_colors);

    const n = rgb_colors.length;
    if (n < 1) {
        console.warn("Требуется хотя бы один цвет, использую исходное изображение.");
        ctx.putImageData(imageData, 0, 0);
        console.timeEnd('convertImagePalette');
        return tempCanvas;
    }
    const segments = n > 1 ? n - 1 : 1;

    // Обрабатываем пиксели
    console.log('Processing pixels');
    for (let y = 0; y < 1080; y++) {
        for (let x = 0; x < 1920; x++) {
            const idx = (y * 1920 + x) * 4;
            const orig_r = data[idx] / 255;
            const orig_g = data[idx + 1] / 255;
            const orig_b = data[idx + 2] / 255;

            const gray = (orig_r + orig_g + orig_b) / 3;
            const pos = gray * segments;
            const seg = Math.floor(pos);
            const frac = pos - seg;

            let newR, newG, newB;
            if (n > 1 && seg < segments) {
                const c1 = rgb_colors[seg];
                const c2 = rgb_colors[seg + 1];
                newR = c1[0] + frac * (c2[0] - c1[0]);
                newG = c1[1] + frac * (c2[1] - c1[1]);
                newB = c1[2] + frac * (c2[2] - c1[2]);
            } else {
                const c = rgb_colors[n - 1];
                newR = c[0];
                newG = c[1];
                newB = c[2];
            }

            if (gray < black_threshold) {
                const blend_factor_black = (black_threshold - gray) / black_threshold;
                newR = orig_r * blend_factor_black + newR * (1 - blend_factor_black);
                newG = orig_g * blend_factor_black + newG * (1 - blend_factor_black);
                newB = orig_b * blend_factor_black + newB * (1 - blend_factor_black);
            }

            if (gray > white_threshold) {
                const blend_factor_white = (gray - white_threshold) / (1 - white_threshold);
                newR = orig_r * blend_factor_white + newR * (1 - blend_factor_white);
                newG = orig_g * blend_factor_white + newG * (1 - blend_factor_white);
                newB = orig_b * blend_factor_white + newB * (1 - blend_factor_white);
            }

            newR = Math.max(0, Math.min(1, newR)) * 255;
            newG = Math.max(0, Math.min(1, newG)) * 255;
            newB = Math.max(0, Math.min(1, newB)) * 255;

            newData[idx] = Math.round(newR);
            newData[idx + 1] = Math.round(newG);
            newData[idx + 2] = Math.round(newB);
        }
    }

    // Логируем первые пиксели после обработки
    console.log('Первые пиксели (после обработки):',
        `R: ${newData[0]}, G: ${newData[1]}, B: ${newData[2]}, A: ${newData[3]}`);

    // Обновляем данные
    try {
        console.log('Putting image data');
        ctx.putImageData(new ImageData(newData, 1920, 1080), 0, 0);
    } catch (e) {
        console.error('Ошибка записи пиксельных данных:', e.message);
        console.timeEnd('convertImagePalette');
        return null;
    }

    console.log('convertImagePalette completed successfully');
    console.timeEnd('convertImagePalette');
    return tempCanvas;
}

async function blurImage(imageCanvas, blurRadius) {
    console.time('blurImage');
    if (!(imageCanvas instanceof HTMLCanvasElement)) {
        console.error('Неверный формат imageCanvas: ожидается HTMLCanvasElement');
        console.timeEnd('blurImage');
        return null;
    }
    if (blurRadius < 0 || blurRadius > 20) {
        console.warn(`Радиус размытия ${blurRadius}px вне допустимого диапазона (0-20), использую 0`);
        blurRadius = 0;
    }

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 1920;
    tempCanvas.height = 1080;
    const ctx = tempCanvas.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = false;

    ctx.filter = `blur(${blurRadius}px)`;
    ctx.drawImage(imageCanvas, 0, 0, 1920, 1080);
    ctx.filter = 'none';

    console.timeEnd('blurImage');
    return tempCanvas;
}

async function processEmployeeData(jsonData, image, rectX = 50, rectY = 50, fontSize = 30, blurRadius = 0, fitMode = 'fit') {
    console.time('processEmployeeData');
    if (!(image instanceof HTMLImageElement)) {
        console.error('Неверный формат изображения: ожидается HTMLImageElement');
        console.timeEnd('processEmployeeData');
        return null;
    }

    const validPrivacyLevels = ['low', 'medium', 'high'];
    if (!validPrivacyLevels.includes(jsonData.employee.privacy_level)) {
        console.warn(`Недопустимый privacy_level: ${jsonData.employee.privacy_level}, использую 'low'`);
        jsonData.employee.privacy_level = 'low';
    }
    console.log('Processing with privacy_level:', jsonData.employee.privacy_level);

    const employee = jsonData.employee;
    const privacyLevel = employee.privacy_level;
    let lines = [];
    if (privacyLevel === 'low') {
        lines = [
            employee.full_name,
            employee.position
        ];
    } else if (privacyLevel === 'medium') {
        lines = [
            employee.full_name,
            employee.position,
            employee.company,
            employee.department,
            employee.office_location
        ];
    } else if (privacyLevel === 'high') {
        lines = [
            employee.full_name,
            employee.position,
            employee.company,
            employee.department,
            employee.office_location,
            `Email: ${employee.contact.email}`,
            `Telegram: ${employee.contact.telegram}`,
            `Slogan: ${employee.branding.slogan}`
        ];
    }

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 1920;
    tempCanvas.height = 1080;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.font = `${fontSize}px Arial, sans-serif`;
    console.log('Font set for tempCtx:', tempCtx.font);

    const lineHeight = 50;
    const paddingX = 20;
    const paddingY = 2;
    const maxWidth = Math.max(...lines.map(line => tempCtx.measureText(line).width));
    const rectWidth = maxWidth + paddingX * 2;
    const rectHeight = lineHeight - 10 + paddingY;
    const cornerRadius = 10;

    if (rectX + rectWidth > 1920 || rectY + rectHeight * lines.length > 1080) {
        console.warn('Текст выходит за пределы изображения!');
    }

    const canvas = document.createElement('canvas');
    canvas.width = 1920;
    canvas.height = 1080;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = false;

    let processedImage;
    try {
        const hex_colors = [
            jsonData.employee.branding.corporate_colors.primary,
            jsonData.employee.branding.corporate_colors.secondary
        ];
        processedImage = await convertImagePalette(image, hex_colors, 0.37, 0.25, fitMode);
        if (!processedImage) {
            throw new Error('Не удалось обработать изображение');
        }
    } catch (e) {
        console.error('Ошибка преобразования палитры:', e.message);
        ctx.drawImage(image, 0, 0, 1920, 1080);
        processedImage = canvas;
    }

    let finalImage = processedImage;
    if (blurRadius > 0) {
        finalImage = await blurImage(processedImage, blurRadius);
        if (!finalImage) {
            console.error('Ошибка размытия изображения');
            finalImage = processedImage;
        }
    }

    ctx.drawImage(finalImage, 0, 0, 1920, 1080);

    ctx.font = `${fontSize}px Arial, sans-serif`;
    console.log('Font set for ctx:', ctx.font);

    const imageData = ctx.getImageData(rectX, rectY, rectWidth, rectHeight * lines.length);
    const data = imageData.data;
    let r = 0, g = 0, b = 0;
    for (let i = 0; i < data.length; i += 4) {
        r += data[i];
        g += data[i + 1];
        b += data[i + 2];
    }
    const pixelCount = data.length / 4;
    r = Math.round(r / pixelCount) / 255;
    g = Math.round(g / pixelCount) / 255;
    b = Math.round(b / pixelCount) / 255;

    const luminanceBackground = 0.299 * r + 0.587 * g + 0.114 * b;

    let backgroundColor, textColor;
    if (luminanceBackground > 0.5) {
        backgroundColor = 'rgba(0, 0, 0, 0.4)';
        textColor = 'white';
    } else {
        backgroundColor = 'rgba(255, 255, 255, 0.4)';
        textColor = 'black';
    }

    ctx.fillStyle = backgroundColor;
    lines.forEach((_, index) => {
        ctx.beginPath();
        ctx.roundRect(
            rectX,
            rectY + index * lineHeight - 5,
            rectWidth,
            rectHeight,
            cornerRadius
        );
        ctx.fill();
    });

    ctx.fillStyle = textColor;
    ctx.textAlign = 'left';
    lines.forEach((line, index) => {
        const textY = rectY + (index + 0.5) * lineHeight + paddingY / 2;
        ctx.fillText(line, rectX + paddingX, textY);
    });

    console.timeEnd('processEmployeeData');
    return canvas;
}