// Launch the SDK's actual WPE executable with the resource settings from its
// shell wrapper, preserving verified extracted native libraries. This avoids
// an installed-package-cache heuristic; missing actual dependencies still fail.
import {dirname,join} from 'node:path';
import {spawnSync} from 'node:child_process';
import {realpathSync} from 'node:fs';
import assert from 'node:assert/strict';
export function nativeWebkitOptions(engine) {
  const runtime=process.env.JOPE_TS_NATIVE_RUNTIME;
  if(!runtime) return {};
  const root=realpathSync(runtime);
  const libraries=['libGLESv2.so.2','libx264.so.164','gstreamer-1.0/libgstlibav.so'];
  for(const name of libraries) {
    const file=realpathSync(join(root,'usr/lib/x86_64-linux-gnu',name));
    assert.ok(file.startsWith(root+'/'));
    const result=spawnSync('ldd',[file],{encoding:'utf8'});
    assert.equal(result.status,0,result.stderr);
    assert.ok(!result.stdout.includes('not found'),result.stdout);
  }
  const mini=join(dirname(engine.executablePath()),'minibrowser-wpe');
  const executablePath=join(mini,'bin/MiniBrowser');
  const env={...process.env,WEBKIT_EXEC_PATH:join(mini,'bin'),WEBKIT_INJECTED_BUNDLE_PATH:join(mini,'lib'),WEBKIT_INSPECTOR_RESOURCES_PATH:join(mini,'share'),WEBKIT_FORCE_COMPLEX_TEXT:'1',LD_LIBRARY_PATH:[join(mini,'lib'),join(mini,'sys/lib'),process.env.LD_LIBRARY_PATH].join(':')};
  const result=spawnSync('ldd',[executablePath],{env,encoding:'utf8'});
  assert.equal(result.status,0,result.stderr);
  assert.ok(!result.stdout.includes('not found'),result.stdout);
  return {executablePath,env};
}
