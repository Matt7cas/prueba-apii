const express = require('express');
const app = express();
const ytdl = require('ytdl-core');
const ffmpeg = require('fluent-ffmpeg');
const axios = require('axios');
const cheerio = require('cheerio');
const fetch = require('node-fetch');
const multer = require('multer');
const upload = multer({ storage: multer.memoryStorage() });
const Jimp = require('jimp');

// Middleware para JSON
app.use(express.json());

// ---------------------------
// Endpoint 1: ytmp3Download
// GET /mp3?url=<URL_DE_YOUTUBE>
// Responde con el audio en formato MP3 y envía en headers metadatos (título, descripción, thumbnail)
// ---------------------------
app.get('/mp3', async (req, res) => {
  const url = req.query.url;
  if (!url) {
    return res.status(400).json({ error: 'Debe proporcionar una URL' });
  }

  try {
    // Obtiene información del video
    const info = await ytdl.getInfo(url);
    const title = info.videoDetails.title;
    const description = info.videoDetails.description;
    const thumbnails = info.videoDetails.thumbnails;
    const thumbnail = thumbnails[thumbnails.length - 1]?.url || '';

    // Establece los metadatos en los headers
    res.setHeader('X-Titulo', title);
    res.setHeader('X-Descripcion', description);
    res.setHeader('X-Thumbnail', thumbnail);
    res.setHeader('Content-Type', 'audio/mpeg');

    // Inicia la descarga del audio y su conversión a MP3 usando ffmpeg
    const stream = ytdl(url, { quality: 'highestaudio' });
    ffmpeg(stream)
      .audioBitrate(128)
      .format('mp3')
      .on('error', (err) => {
        console.error('Error en ffmpeg:', err.message);
        if (!res.headersSent) res.status(500).json({ error: err.message });
      })
      .pipe(res, { end: true });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ---------------------------
// Endpoint 2: ytmp4Download
// GET /mp4?url=<URL_DE_YOUTUBE>
// Responde con el video en formato MP4 y envía en headers metadatos (título, descripción, thumbnail)
// ---------------------------
app.get('/mp4', async (req, res) => {
  const url = req.query.url;
  if (!url) {
    return res.status(400).json({ error: 'Debe proporcionar una URL' });
  }

  try {
    // Obtiene información del video
    const info = await ytdl.getInfo(url);
    const title = info.videoDetails.title;
    const description = info.videoDetails.description;
    const thumbnails = info.videoDetails.thumbnails;
    const thumbnail = thumbnails[thumbnails.length - 1]?.url || '';

    res.setHeader('X-Titulo', title);
    res.setHeader('X-Descripcion', description);
    res.setHeader('X-Thumbnail', thumbnail);
    res.setHeader('Content-Type', 'video/mp4');

    // Descarga el video (audio y video juntos) en calidad máxima
    ytdl(url, { filter: 'audioandvideo', quality: 'highestvideo' })
      .on('error', (err) => {
        console.error('Error en ytdl:', err.message);
        if (!res.headersSent) res.status(500).json({ error: err.message });
      })
      .pipe(res);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ---------------------------
// Endpoint 3: Acceso a una IA profesional gratis
// POST /ia
// Recibe un JSON { "texto": "..." } y retorna una generación de texto usando el modelo GPT-2 de Hugging Face
// ---------------------------
app.post('/ia', async (req, res) => {
  const { texto } = req.body;
  if (!texto) {
    return res.status(400).json({ error: "Debe enviar un JSON con la clave 'texto'" });
  }

  try {
    // Se utiliza la API pública de Hugging Face para el modelo GPT-2
    const response = await fetch('https://api-inference.huggingface.co/models/gpt2', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputs: texto })
    });
    const result = await response.json();
    if (result.error) {
      return res.status(500).json({ error: result.error });
    }
    // Se asume que el resultado es un arreglo de objetos con la clave generated_text
    res.json({ resultado: result[0].generated_text });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ---------------------------
// Endpoint 4: Imagen a texto (detección de objetos)
// POST /image-to-text
// Recibe una imagen (form-data, campo "imagen"), utiliza el modelo coco-ssd de TensorFlow.js para detectar objetos, 
// anota la imagen y retorna la imagen anotada junto con una descripción en los headers.
// ---------------------------

// Carga el modelo de detección de objetos (coco-ssd) al iniciar la aplicación.
let objectDetectionModel;
(async () => {
  try {
    const cocoSsd = require('@tensorflow-models/coco-ssd');
    objectDetectionModel = await cocoSsd.load();
    console.log('Modelo coco-ssd cargado.');
  } catch (err) {
    console.error('Error al cargar coco-ssd:', err);
  }
})();

app.post('/image-to-text', upload.single('imagen'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "Debe subir una imagen con el parámetro 'imagen'" });
  }

  try {
    // Carga la imagen con Jimp
    const image = await Jimp.read(req.file.buffer);
    // Convierte la imagen a un buffer JPEG
    const imageBuffer = await image.getBufferAsync(Jimp.MIME_JPEG);

    // Decodifica la imagen en un tensor (la función detect de coco-ssd espera un tensor de forma [H, W, 3])
    const tf = require('@tensorflow/tfjs-node');
    const tfImage = tf.node.decodeImage(imageBuffer);

    // Realiza la detección
    const predictions = await objectDetectionModel.detect(tfImage);
    tfImage.dispose();

    // Se filtran las detecciones con un umbral mínimo de confianza
    const threshold = 0.8;
    const detections = predictions.filter(pred => pred.score >= threshold);
    const detectedClasses = detections.map(pred => pred.class);
    const uniqueObjects = Array.from(new Set(detectedClasses));
    const description = uniqueObjects.length > 0
      ? "Objetos detectados: " + uniqueObjects.join(', ')
      : "No se detectaron objetos con alta confianza.";

    // Anota la imagen: dibuja un rectángulo y escribe la etiqueta para cada detección
    // Debido a las limitaciones de Jimp para dibujar, se realiza un trazado simple de líneas para el bounding box.
    const font = await Jimp.loadFont(Jimp.FONT_SANS_16_BLACK);
    detections.forEach(pred => {
      const [x, y, width, height] = pred.bbox.map(v => Math.floor(v));
      // Dibuja líneas horizontales
      for (let i = x; i < x + width; i++) {
        image.setPixelColor(Jimp.cssColorToHex('#FF0000'), i, y);
        image.setPixelColor(Jimp.cssColorToHex('#FF0000'), i, y + height);
      }
      // Dibuja líneas verticales
      for (let j = y; j < y + height; j++) {
        image.setPixelColor(Jimp.cssColorToHex('#FF0000'), x, j);
        image.setPixelColor(Jimp.cssColorToHex('#FF0000'), x + width, j);
      }
      // Escribe la etiqueta
      image.print(font, x, y, pred.class);
    });

    // Prepara la imagen anotada para enviar
    const annotatedBuffer = await image.getBufferAsync(Jimp.MIME_JPEG);
    res.setHeader('X-Descripcion', description);
    res.set('Content-Type', 'image/jpeg');
    res.send(annotatedBuffer);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ---------------------------
// Endpoint 5: Scraping a chochox.com
// GET /chochox?q=<término de búsqueda>
// Realiza una búsqueda en chochox.com, extrae el primer resultado y recopila todas las imágenes de esa página.
// ---------------------------
app.get('/chochox', async (req, res) => {
  const query = req.query.q;
  if (!query) {
    return res.status(400).json({ error: "Debe proporcionar el parámetro de búsqueda 'q'" });
  }

  try {
    // Se asume que la búsqueda se realiza en esta URL; ajústala según la estructura real del sitio.
    const searchUrl = `https://chochox.com/search?q=${encodeURIComponent(query)}`;
    const response = await axios.get(searchUrl, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    const $ = cheerio.load(response.data);

    // Se asume que cada resultado está en un div con clase "result" (ajusta el selector según la web actual)
    const firstResult = $('.result').first();
    if (!firstResult.length) {
      return res.status(404).json({ error: "No se encontró ningún resultado" });
    }
    const enlace = firstResult.find('a').attr('href');
    const titulo = firstResult.text().trim();

    // Se accede a la página del primer resultado para extraer imágenes
    const detailResponse = await axios.get(enlace, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    const $detail = cheerio.load(detailResponse.data);
    const imagenes = [];
    $detail('img').each((i, el) => {
      const src = $detail(el).attr('src');
      if (src) imagenes.push(src);
    });

    res.json({
      primer_resultado: { titulo, enlace },
      imagenes
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ---------------------------
// Inicio de la aplicación
// ---------------------------
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`API corriendo en el puerto ${PORT}`);
});
