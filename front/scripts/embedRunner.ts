import { storePdfsInPinecone } from "../src/app/lib/embedAndStore.js";

(async () => {
  await storePdfsInPinecone();
})();
