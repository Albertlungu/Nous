# macOS Trusted Release (Signed + Notarized)

This project is configured to notarize automatically during `electron-builder` after signing.

## Prerequisites

1. Apple Developer membership.
2. A valid `Developer ID Application` certificate installed in Keychain on this Mac.
3. App-specific password for your Apple ID.

## One-time credential setup (shell env)

Set these in your shell before building:

```bash
export APPLE_ID="your-apple-id@example.com"
export APPLE_APP_SPECIFIC_PASSWORD="xxxx-xxxx-xxxx-xxxx"
export APPLE_TEAM_ID="YOURTEAMID"
```

Optional (if you have multiple certificates):

```bash
export CSC_NAME="Developer ID Application: Your Name (TEAMID)"
```

## Build a release DMG

```bash
cd electron-app
npm run build:mac:release
```

This performs:
1. Build a standalone backend binary (`scripts/build_backend_macos.sh`)
2. Developer ID signing (electron-builder)
3. Notarization (`scripts/notarize.js` via `@electron/notarize`)
4. DMG creation in `electron-app/dist`

The release no longer relies on shipping a machine-specific Python `venv` inside the app bundle.

## Validate locally

```bash
spctl -a -vvv "electron-app/dist/mac-arm64/Nous.app"
```

Expected: accepted by Gatekeeper.

## Publish

Upload the generated DMG to GitHub Releases.
If notarization succeeded, downloaded copies should open without the "damaged" error.
