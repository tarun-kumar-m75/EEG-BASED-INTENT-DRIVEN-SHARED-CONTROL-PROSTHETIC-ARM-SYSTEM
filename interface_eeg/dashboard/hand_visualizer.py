"""
dashboard/hand_visualizer.py
============================
Clean, lightweight vector SVG visualizer for the prosthetic hand.
Renders smooth kinematic transitions between OPEN and CLOSED states
following clinical biomedical engineering design aesthetics.
"""

def get_hand_svg(arm_state: str = "OPEN") -> str:
    """
    Generates an animated vector SVG of the prosthetic hand.
    arm_state: 'OPEN' or 'CLOSED'.
    """
    is_closed = (arm_state == "CLOSED")
    
    # Restrained Clinical Palette:
    accent_color = "#38A169" if is_closed else "#3AAFA9"
    status_text = "GRASP (CLOSED)" if is_closed else "RELEASE (OPEN)"
    grip_angle = "90° (GRIP ACTIVE)" if is_closed else "0° (EXTENDED)"
    
    # Finger coordinates: Open vs Curled Closed
    if is_closed:
        f_index_y = 110
        f_mid_y = 105
        f_ring_y = 108
        f_pinky_y = 115
        thumb_x = 90
        thumb_y = 125
        knuckle_color = "#38A169"
    else:
        f_index_y = 38
        f_mid_y = 30
        f_ring_y = 34
        f_pinky_y = 48
        thumb_x = 45
        thumb_y = 95
        knuckle_color = "#4B5563"

    # Important: Absolutely zero leading indentation on lines to prevent markdown code-block conversion
    lines = [
        '<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; background-color:#1B1F25; border:1px solid #30343B; border-radius:4px; padding:10px; margin:0 auto; max-width:240px;">',
        f'<div style="font-family:\'Segoe UI\', Arial, sans-serif; font-size:12px; font-weight:600; color:#E5E7EB; margin-bottom:2px;">Actuator: <span style="color:{accent_color};">{status_text}</span></div>',
        f'<div style="font-family:\'Consolas\', monospace; font-size:11px; color:#9CA3AF; margin-bottom:6px;">Position: {grip_angle}</div>',
        '<svg width="170" height="170" viewBox="0 0 200 200" style="transition:all 0.35s ease-in-out;">',
        '<rect x="78" y="155" width="44" height="35" rx="3" fill="#14171D" stroke="#30343B" stroke-width="1.5"/>',
        '<rect x="62" y="95" width="76" height="58" rx="4" fill="#181B21" stroke="#30343B" stroke-width="1.5"/>',
        f'<circle cx="100" cy="124" r="13" fill="#101216" stroke="{accent_color}" stroke-width="1.5"/>',
        '<text x="100" y="128" font-family="monospace" font-size="8" fill="#9CA3AF" text-anchor="middle" font-weight="600">PWM</text>',
        f'<circle cx="74" cy="95" r="3.5" fill="{knuckle_color}"/>',
        f'<circle cx="91" cy="93" r="3.5" fill="{knuckle_color}"/>',
        f'<circle cx="109" cy="93" r="3.5" fill="{knuckle_color}"/>',
        f'<circle cx="126" cy="95" r="3.5" fill="{knuckle_color}"/>',
        f'<line x1="74" y1="95" x2="74" y2="{f_index_y}" stroke="{accent_color}" stroke-width="5" stroke-linecap="round"/>',
        f'<circle cx="74" cy="{f_index_y}" r="2.5" fill="#E5E7EB"/>',
        f'<line x1="91" y1="93" x2="91" y2="{f_mid_y}" stroke="{accent_color}" stroke-width="5.5" stroke-linecap="round"/>',
        f'<circle cx="91" cy="{f_mid_y}" r="2.5" fill="#E5E7EB"/>',
        f'<line x1="109" y1="93" x2="109" y2="{f_ring_y}" stroke="{accent_color}" stroke-width="5" stroke-linecap="round"/>',
        f'<circle cx="109" cy="{f_ring_y}" r="2.5" fill="#E5E7EB"/>',
        f'<line x1="126" y1="95" x2="126" y2="{f_pinky_y}" stroke="{accent_color}" stroke-width="4.5" stroke-linecap="round"/>',
        f'<circle cx="126" cy="{f_pinky_y}" r="2" fill="#E5E7EB"/>',
        f'<circle cx="62" cy="120" r="3.5" fill="{knuckle_color}"/>',
        f'<line x1="62" y1="120" x2="{thumb_x}" y2="{thumb_y}" stroke="{accent_color}" stroke-width="5.5" stroke-linecap="round"/>',
        f'<circle cx="{thumb_x}" cy="{thumb_y}" r="2.5" fill="#E5E7EB"/>',
        '</svg>',
        '</div>'
    ]
    return "".join(lines)
