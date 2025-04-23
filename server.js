const express = require("express");
const cors = require("cors");
const multer = require("multer");
const { execFile } = require("child_process");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = 5000;

// Enable CORS for your frontend origin
app.use(
  cors({
    origin: "https://frontend-plant.vercel.app",
  })
);

const upload = multer({ dest: "uploads/" });

app.post("/api/upload", upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image uploaded." });
  }

  const imagePath = path.join(__dirname, req.file.path);

  execFile("python", ["inference.py", imagePath], (error, stdout, stderr) => {
    // Clean up uploaded file
    fs.unlinkSync(imagePath);

    if (error) {
      console.error("Inference error:", stderr);
      return res
        .status(500)
        .json({ error: "Inference failed.", details: stderr });
    }

    try {
      const result = JSON.parse(stdout);
      res.json(result);
    } catch (e) {
      console.error("Invalid JSON response from Python script:", stdout);
      res.status(500).json({ error: "Invalid response from Python script." });
    }
  });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
