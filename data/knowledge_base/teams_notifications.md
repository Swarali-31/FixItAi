# Microsoft Teams notifications are not arriving

Category: App troubleshooting
Technologies: Microsoft Teams, Windows, macOS, iOS, Android, notifications

If Teams notifications are missing, check the app, operating system, account, and focus settings in this order. Menu labels can vary slightly by Teams version and device.

## Teams desktop or web

1. In Teams, open Settings and more, then Settings, then Notifications and activity. Check that notifications for the affected activity (chat, channel, mentions, or meetings) are enabled and are set to appear where expected. Microsoft's current guide: https://support.microsoft.com/en-us/teams/notifications-settings/manage-notifications-in-microsoft-teams
2. If only one chat or channel is affected, open that conversation's or channel's notification settings and check that it is not muted. For channels, use More options next to the channel, then Channel notifications. Guide: https://support.microsoft.com/en-us/teams/teams-channels/manage-channel-notifications-in-microsoft-teams
3. Check the operating system's notification permissions for Teams. On Windows, open Settings > System > Notifications, make sure notifications are enabled, and check Do not disturb / focus schedules. Guide: https://support.microsoft.com/en-us/windows/notifications-and-do-not-disturb-in-windows
4. Check whether another signed-in Teams account or organization is selected and allowed to send notifications. Guide: https://support.microsoft.com/en-us/teams/platform/manage-accounts-and-organizations-in-microsoft-teams
5. Quit and reopen Teams. If the problem is limited to one device, compare notification settings on another device before resetting or reinstalling the app.

## Teams mobile

1. In Teams, tap your profile picture > Notifications. Check quiet time / block notifications, in-app notifications, and whether notifications are blocked while using Teams on another device.
2. In iOS or Android settings, allow notifications for Teams and check Focus, Do Not Disturb, battery-saving, and background-app restrictions.
3. If the Teams mobile app is not listed in iOS notification settings, Microsoft's troubleshooting guide recommends reinstalling Teams. Check the official mobile troubleshooting steps: https://support.microsoft.com/en-us/teams/notifications-settings/troubleshoot-notifications-in-microsoft-teams-mobile-apps

## Narrow down the cause

- No notifications from any app: check operating-system Do Not Disturb / Focus and notification permissions first.
- Other apps notify normally but Teams does not: check Teams notification settings and the affected chat/channel mute state.
- Notifications appear on one device but not another: check that device's OS permissions, Focus / battery restrictions, and Teams quiet-time settings.
- Notifications arrive late: check network access and background-app restrictions; on managed work devices, ask the organization administrator whether policy restricts notifications.

These steps are general checks, not a diagnosis of a specific tenant or device policy. Confirm menu names and current behavior in Microsoft's official documentation for your device.
