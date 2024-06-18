const auth = (req, res, next) => {
  if (req.query["user"] === "azw") {
    return next();
  }
  throw new Error("Invalid user");
};

module.exports = { auth };
