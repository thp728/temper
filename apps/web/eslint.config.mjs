import nextConfig from "eslint-config-next";

// eslint-config-next ships flat configs directly in Next 16: an array of
// [core-web-vitals, typescript, ...]. The generated client is excluded --
// it is a build artifact, reviewed as the contract diff, not linted here.
const eslintConfig = [
  ...nextConfig,
  {
    ignores: [
      ".next/**",
      "node_modules/**",
      "src/lib/api/generated/**",
    ],
  },
];

export default eslintConfig;
