const express = require("express");

const { auth } = require("./middleware.js");
const router = require("./router");

const PORT = 5000;

const app = express();

// middleware if applicable for all requests
// app.use(auth);

app.use("", router);

app.listen(PORT, () => {
  console.log(`Listening on port ${PORT}`);
});
