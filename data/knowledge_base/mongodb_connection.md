# MongoDB Connection Timeout on Port 27017
Category: NoSQL Databases & Document Stores
Technologies: MongoDB, Node.js, Python, Mongoose, PyMongo

When connecting to MongoDB from a web service:
`MongooseServerSelectionError: connect ECONNREFUSED 127.0.0.1:27017` or `pymongo.errors.ServerSelectionTimeoutError: localhost:27017: [Errno 111] Connection refused`.

## Root Cause
1. The MongoDB daemon (`mongod`) is not running.
2. In MongoDB 5.0+, the service binds by default only to `127.0.0.1`.
3. In cloud environments (Atlas), your active client public IP is not whitelisted in Network Access.

## Resolution Steps
1. Verify if `mongod` is active:
   `sudo systemctl status mongod` or `brew services list`
2. Start the service:
   `sudo systemctl start mongod` or `brew services start mongodb-community`
3. Edit `/etc/mongod.conf` to check net configuration:
   `net.bindIp: 127.0.0.1`
4. If using MongoDB Atlas, visit Atlas Console -> Security -> Network Access -> Add Current IP Address.
