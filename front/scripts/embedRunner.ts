//front/scripts/embedRunner.ts
import "dotenv/config";
import { storePdfsInPinecone } from "../src/app/lib/embedAndStore.js";

(async () => {
  await storePdfsInPinecone();
})();
