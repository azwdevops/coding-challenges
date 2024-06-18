const express = require("express");
const { auth } = require("./middleware");
const { home } = require("./controller");

const router = express.Router();

router.get("/", auth, home);

module.exports = router;
