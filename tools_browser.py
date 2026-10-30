"""Headless browser automation (Playwright), degrading gracefully.

Copilot-style agents verify front-end work by LOOKING at it. These tools give
the harness the same ability: open a page, snapshot its interactive elements
with numbered refs, click/type/navigate, and screenshot -- where the
screenshot returns a multimodal envelope (see tools_media) so the model can
inspect the pixels rather than guess from the DOM.

Playwright is an OPTIONAL dependency (requirements-browser.txt). When it is
absent every tool returns the same clear NOT AVAILABLE message instead of a
traceback, so a slim container image keeps working and the model knows exactly
what to tell the operator.

One browser per process, one page at a time: the harness runs one task per
process, and a pool of tabs would only multiply the ways a run can leak a
chromium process. ``browser_close`` releases everything.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Tuple

import config
import tools_media

#: Elements worth a ref: the things an agent clicks or types into.
_SELECTOR = ("a, button, input, select, textarea, [role='button'], "
             "[onclick], [tabindex]")

_STATE: dict = {"pw": None, "browser": None, "page": None,
                "refs": {}, "next": 1}

_UNAVAILABLE = (
    "NOT AVAILABLE: browser automation needs Playwright. Install it with "
    "`pip install -r requirements-browser.txt` and then "
    "`playwright install chromium`. Nothing was opened.")


def _page() -> Tuple[Optional[Any], Optional[str]]:
    """The live page, launching chromium on first use. (page, error)."""
    page = _STATE.get("page")
    if page is not None:
        try:
            if not page.is_closed():
                return page, None
        except Exception:  # noqa: BLE001
            pass
        _STATE["page"] = None
    try:
        from playwright.sync_api import sync_playwright
    except Exception:  # noqa: BLE001  (ImportError or a broken install)
        return None, _UNAVAILABLE
    try:
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=config.BROWSER_HEADLESS)
        page = browser.new_page()
        page.set_default_timeout(config.BROWSER_TIMEOUT_MS)
    except Exception as exc:  # noqa: BLE001
        return None, (f"BROWSER LAUNCH FAILED: {type(exc).__name__}: {exc}. "
                      f"Run `playwright install chromium` if the browser "
                      f"binary is missing.")
    _STATE.update(pw=pw, browser=browser, page=page, refs={}, next=1)
    return page, None


def _mini_state(page: Any) -> str:
    try:
        return f"url={page.url}\ntitle={page.title()}"
    except Exception:  # noqa: BLE001
        return "(page state unreadable)"


def _locate(page: Any, ref: Optional[str], selector: Optional[str],
            text: Optional[str]):
    """ref > selector > visible text. Returns (locator_or_handle, error)."""
    if ref:
        handle = _STATE["refs"].get(ref)
        if handle is None:
            return None, (f"UNKNOWN REF {ref}; refs are rebuilt by every "
                          f"browser_snapshot -- take a fresh one.")
        return handle, None
    if selector:
        return page.locator(selector).first, None
    if text:
        return page.get_by_text(text, exact=False).first, None
    return None, "Give one of ref, selector or text."


def _describe(el: Any) -> str:
    try:
        tag = el.evaluate("e => e.tagName.toLowerCase()")
    except Exception:  # noqa: BLE001
        tag = "?"
    bits = [f"<{tag}>"]
    try:
        if tag in ("input", "select", "textarea"):
            bits.append(f"type={el.get_attribute('type') or ''}"
                        f" name={el.get_attribute('name') or ''}"
                        f" placeholder={el.get_attribute('placeholder') or ''}")
        else:
            text = " ".join((el.inner_text() or "").split())[:80]
            if text:
                bits.append(f'"{text}"')
        href = el.get_attribute("href")
        if href:
            bits.append(f"href={href[:80]}")
    except Exception:  # noqa: BLE001
        pass
    return " ".join(b for b in bits if b)


def browser_open(url: str) -> str:
    """Launch (or reuse) the headless browser and load ``url``.

    Returns the first snapshot, so one call is usually enough to start
    working. Use http(s) or file:// URLs.
    """
    page, err = _page()
    if err:
        return err
    try:
        page.goto(url)
    except Exception as exc:  # noqa: BLE001
        return f"NAVIGATION FAILED for {url}: {type(exc).__name__}: {exc}"
    return f"OPENED {url}\n" + browser_snapshot()


def browser_snapshot() -> str:
    """Numbered interactive elements plus a bounded excerpt of page text.

    Refs (e1, e2, ...) are rebuilt on EVERY snapshot and stay valid until the
    next one -- click/type by ref instead of guessing selectors. If the page
    changed since your last snapshot, take a new one.
    """
    page, err = _page()
    if err:
        return err
    _STATE["refs"] = {}
    _STATE["next"] = 1
    lines = []
    try:
        elements = page.query_selector_all(_SELECTOR)
    except Exception as exc:  # noqa: BLE001
        return f"SNAPSHOT FAILED: {type(exc).__name__}: {exc}"
    for el in elements[:config.BROWSER_MAX_REFS]:
        try:
            if not el.is_visible():
                continue
        except Exception:  # noqa: BLE001
            continue
        ref = f"e{_STATE['next']}"
        _STATE["next"] += 1
        _STATE["refs"][ref] = el
        lines.append(f"[{ref}] {_describe(el)}")
    try:
        body = page.inner_text("body") or ""
    except Exception:  # noqa: BLE001
        body = ""
    excerpt = body[:config.BROWSER_SNAPSHOT_CHARS]
    more = (f"\n... [{len(body) - config.BROWSER_SNAPSHOT_CHARS} more chars; "
            f"use browser_read for the rest]"
            if len(body) > config.BROWSER_SNAPSHOT_CHARS else "")
    return (_mini_state(page) + "\n" + "\n".join(lines)
            + "\n--- page text ---\n" + excerpt + more)


def browser_click(ref: str = "", selector: str = "", text: str = "") -> str:
    """Click an element by snapshot ref, CSS selector, or visible text."""
    page, err = _page()
    if err:
        return err
    target, err = _locate(page, ref or None, selector or None, text or None)
    if err:
        return err
    try:
        target.click()
    except Exception as exc:  # noqa: BLE001
        return f"CLICK FAILED: {type(exc).__name__}: {exc}"
    return f"CLICKED {ref or selector or text!r}\n" + _mini_state(page)


def browser_type(ref: str = "", selector: str = "", text: str = "",
                 submit: bool = False, clear: bool = True) -> str:
    """Type into an input. ``clear`` replaces existing content; ``submit``
    presses Enter afterwards (forms that need a click: use browser_click)."""
    page, err = _page()
    if err:
        return err
    target, err = _locate(page, ref or None, selector or None, None)
    if err:
        return err
    try:
        if clear:
            target.fill(text)
        else:
            target.press_sequentially(text)
        if submit:
            target.press("Enter")
    except Exception as exc:  # noqa: BLE001
        return f"TYPE FAILED: {type(exc).__name__}: {exc}"
    return (f"TYPED {len(text)} char(s) into {ref or selector}"
            + (" + Enter" if submit else "") + "\n" + _mini_state(page))


def browser_navigate(target: str) -> str:
    """Go to a URL, or use 'back', 'forward', 'reload'."""
    page, err = _page()
    if err:
        return err
    try:
        if target == "back":
            page.go_back()
        elif target == "forward":
            page.go_forward()
        elif target == "reload":
            page.reload()
        else:
            page.goto(target)
    except Exception as exc:  # noqa: BLE001
        return f"NAVIGATION FAILED: {type(exc).__name__}: {exc}"
    return _mini_state(page)


def browser_read(max_chars: int = 0) -> str:
    """The page's full visible text (bounded). Use when the snapshot excerpt
    is not enough -- reading an article, checking an error message verbatim."""
    page, err = _page()
    if err:
        return err
    try:
        body = page.inner_text("body") or ""
    except Exception as exc:  # noqa: BLE001
        return f"READ FAILED: {type(exc).__name__}: {exc}"
    limit = int(max_chars) or 20000
    return body[:limit] + ("\n... [truncated]" if len(body) > limit else "")


def browser_screenshot(path: str = "") -> str:
    """Capture the viewport and ATTACH it so you can see the rendered page.

    Returns an image envelope (the loop turns it into a multimodal message);
    the PNG is also saved under sessions/ (or ``path``) for the record.
    """
    page, err = _page()
    if err:
        return err
    dest = Path(path) if path else (
        config.SESSIONS_DIR / f"browser_{int(time.time())}.png")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(dest), full_page=False)
        data = dest.read_bytes()
    except Exception as exc:  # noqa: BLE001
        return f"SCREENSHOT FAILED: {type(exc).__name__}: {exc}"
    return tools_media.encode_image_bytes(
        data, "image/png", str(dest), note="browser screenshot")


def browser_close() -> str:
    """Release the browser and every handle. Call it when done browsing so
    the chromium process does not outlive the task."""
    page, browser, pw = (_STATE.get("page"), _STATE.get("browser"),
                         _STATE.get("pw"))
    for obj, stop in ((page, "close"), (browser, "close"), (pw, "stop")):
        try:
            getattr(obj, stop)()
        except Exception:  # noqa: BLE001
            pass
    _STATE.update(pw=None, browser=None, page=None, refs={}, next=1)
    return "browser closed"
