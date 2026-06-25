import { $ } from "bun";
import fs from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";

const scriptDir = import.meta.dir;
const repoRoot = path.resolve(scriptDir, "..");

try {
  const logPath = path.join(repoRoot, 'mainrun', 'logs', 'mainrun.log')
  if (existsSync(logPath)) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5)
    const newLogPath = path.join(repoRoot, 'mainrun', 'logs', `mainrun_${timestamp}.log`)
    await fs.rename(logPath, newLogPath)
    console.log(`Moved existing log to: mainrun_${timestamp}.log`)
  }

  await $`git -C ${repoRoot} add .`

  const status = await $`git -C ${repoRoot} status --porcelain`
  if (status.stdout.trim() === '') {
    console.log('No changes to checkpoint')
    process.exit(0)
  }

  await $`git -C ${repoRoot} commit -m "Mainrun auto checkpoint"`
  console.log('Auto checkpoint created')
} catch (error) {
  if (error.message.includes('nothing to commit')) {
    console.log('No changes to checkpoint')
    process.exit(0)
  }

  console.error('Failed to create checkpoint:', error.message)
  process.exit(1)
}
