-- Live-verified Omarchy 4.5.0 Hyprland Lua monitor and touch reference for
-- ASUS ZenBook Pro Duo UX582HS (Intel TigerLake-H iGPU 0000:00:02.0 driving
-- both displays while NVIDIA RTX 3080 0000:01:00.0 stays at zero desktop
-- processes).
--
-- Copy into ~/.config/hypr/monitors.lua (after backing up the existing file)
-- and validate with: hyprctl reload && hyprctl configerrors

-- Fallback rule for hot-plugged external monitors.
local omarchy_monitor_scale = "auto"
hl.monitor({ output = "", mode = "preferred", position = "auto", scale = omarchy_monitor_scale })

-- Upper primary 15.6" 4K OLED (3840x2160 @ scale 2 -> 1920x1080 logical) at 0x0.
hl.monitor({ output = "eDP-1", mode = "3840x2160@60", position = "0x0", scale = 2 })

-- Lower 14" matte IPS ScreenPad Plus (3840x1100 @ scale 2 -> 1920x550 logical)
-- stacked directly below eDP-1 at 0x1080 for seamless vertical cursor motion.
hl.monitor({ output = "DP-3", mode = "3840x1100@60", position = "0x1080", scale = 2 })

-- Per-panel ELAN I2C touch and stylus digitizer binding (verified live via
-- `hyprctl devices -j` on UX582HS under Omarchy 4.5.0).
hl.device({ name = "elan9008:00-04f3:2c56", output = "eDP-1" })
hl.device({ name = "elan9008:00-04f3:2c56-stylus", output = "eDP-1" })
hl.device({ name = "elan9009:00-04f3:2c23", output = "DP-3" })
hl.device({ name = "elan9009:00-04f3:2c23-stylus", output = "DP-3" })

-- Whole-number GDK_SCALE matching monitor scale = 2 for crisp XWayland apps.
local omarchy_gdk_scale = 2
hl.env("GDK_SCALE", tostring(omarchy_gdk_scale))
