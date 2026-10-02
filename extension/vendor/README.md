# Local Frontend Dependencies

These pinned upstream distributions run locally in the Chrome extension. No
CDN script, API endpoint, remote font, or runtime package installation is used.

| File | Upstream Package | Version | Purpose |
| --- | --- | --- | --- |
| `lucide.min.js` | `lucide` | 1.49.0 | Accessible interface icons |
| `marked.umd.js` | `marked` | 18.0.14 | Markdown presentation of existing answer text |
| `purify.min.js` | `dompurify` | 3.4.16 | Sanitization of rendered Markdown |

Distributions were obtained from the versioned npm packages through unpkg.
Upstream license files are included in this directory without modification.
Raw answer HTML is escaped before sanitization. Source and evidence values are
inserted as text, and only HTTP/HTTPS URLs are clickable.
