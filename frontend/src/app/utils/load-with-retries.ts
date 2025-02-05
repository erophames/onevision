export function loadWithRetries(importer: () => Promise<any>): () => Promise<any> {
  const retryImport = async () => {
    try {
      return await importer();
    } catch (error: any) {
      for (let i = 0; i < 5; i++) {
        await new Promise(resolve => setTimeout(resolve, 1000 * 2 ** i));

        try {
          const errorMessage = error.message;
          const url = errorMessage.replace('Failed to fetch dynamically imported module: ', '').trim();

          if (url) {
            const cacheBustedUrl = `${url}?t=${+new Date()}`;
            return await import(cacheBustedUrl);
          }
        } catch (e) {
          console.log('Retrying import');
        }
      }
      throw error;
    }
  };

  return retryImport;
}
