export default {
  async scheduled(event, env, ctx) {
    console.log("PROBE fired: " + event.cron + " at " + new Date().toISOString());
  },
  async fetch() { return new Response("probe alive\n"); },
};
