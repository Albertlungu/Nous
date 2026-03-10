const path = require('path');
const { notarize } = require('@electron/notarize');

exports.default = async function notarizeApp(context) {
    if (process.platform !== 'darwin') {
        return;
    }

    const { electronPlatformName, appOutDir, packager } = context;
    if (electronPlatformName !== 'darwin') {
        return;
    }

    const appleId = process.env.APPLE_ID;
    const appleAppSpecificPassword = process.env.APPLE_APP_SPECIFIC_PASSWORD;
    const appleTeamId = process.env.APPLE_TEAM_ID;

    if (!appleId || !appleAppSpecificPassword || !appleTeamId) {
        console.log('Skipping notarization: APPLE_ID / APPLE_APP_SPECIFIC_PASSWORD / APPLE_TEAM_ID not fully set.');
        return;
    }

    const appName = packager.appInfo.productFilename;
    const appPath = path.join(appOutDir, `${appName}.app`);

    console.log(`Notarizing ${appPath}...`);
    await notarize({
        appBundleId: packager.appInfo.id,
        appPath,
        appleId,
        appleIdPassword: appleAppSpecificPassword,
        teamId: appleTeamId,
        tool: 'notarytool'
    });
    console.log('Notarization completed.');
};
