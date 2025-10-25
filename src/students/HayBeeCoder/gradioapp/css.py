UI_CSS = """
        /* Modern Container Styling */
        .gradio-container {
            min-height: 100vh;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            width: 100%;
            box-sizing: border-box;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
        }
        p{
            font-size: 0.75rem !important;
        }
        .gradio-container main {
            padding: 1.5rem !important;
        }
        /* Dark mode image upload styling */
        .gradio-container [data-testid="image"] {
            background: rgba(30, 30, 50, 0.8) !important;
            border: 2px solid rgba(102, 126, 234, 0.5) !important;
        }
        /* Compact upload area */
        .gradio-container .upload-wrap {
            background: rgba(30, 30, 50, 0.8) !important;
            border: 2px dashed rgba(102, 126, 234, 0.5) !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            text-align: center !important;
            min-height: 150px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        .gradio-container .upload-wrap:hover {
            background: rgba(40, 40, 60, 0.9) !important;
            border-color: #667eea !important;
        }
        /* Upload text styling for dark mode */
        .gradio-container .upload-wrap .text-sm {
            color: #e2e8f0 !important;
        }
        /* Make info text smaller */
        .gradio-container .text-sm {
            font-size: 0.75rem !important;
            line-height: 1.2 !important;
        }
        .gradio-container .form .text-sm {
            font-size: 0.7rem !important;
            margin-top: 0.25rem !important;
        }

        /* Target Gradio's info text specifically */
        .gradio-container [data-testid="form"] .text-sm,
        .gradio-container [data-testid="slider"] .text-sm,
        .gradio-container [data-testid="checkbox"] .text-sm,
        .gradio-container [data-testid="dropdown"] .text-sm,
        .gradio-container [data-testid="textbox"] .text-sm {
            font-size: 0.7rem !important;
            color: var(--body-text-color-secondary) !important;
            margin-top: 0.25rem !important;
        }

        /* Force responsive behavior with more specific selectors */
        @media (max-width: 768px) {
            /* Override Gradio's default styles more aggressively */
            .gradio-container * {
                box-sizing: border-box !important;
            }

            /* Force all rows to be single column on mobile */
            .gradio-container > div > div {
                display: flex !important;
                flex-direction: column !important;
                width: 100% !important;
            }

            /* Make all form elements full width on mobile */
            .gradio-container input,
            .gradio-container select,
            .gradio-container textarea,
            .gradio-container button {
                width: 100% !important;
                max-width: 100% !important;
                margin: 0.25rem 0 !important;
            }

            /* Make info text even smaller on mobile */
            .gradio-container .text-sm {
                font-size: 0.65rem !important;
            }

            /* Stack all accordion content vertically */
            .gradio-container [role="region"] {
                display: flex !important;
                flex-direction: column !important;
            }
        }

        /* Main App Container */
        .gradio-container > div {
            background: rgba(30, 30, 50, 0.95) !important;
            border-radius: 24px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(102, 126, 234, 0.3);
            overflow: hidden;
            animation: slideInUp 0.8s ease-out;
        }

        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Header Styling */
        .gradio-container h1 {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            margin: 2rem 0;
            letter-spacing: -0.02em;
            animation: fadeInScale 1s ease-out 0.2s both;
        }

        @keyframes fadeInScale {
            from {
                opacity: 0;
                transform: scale(0.9);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }

        /* Subtitle Styling */
        .gradio-container p {
            text-align: center;
            color: #f0f0f0;
            font-size: 1.1rem;
            margin-bottom: 3rem;
            animation: fadeIn 1s ease-out 0.4s both;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Upload Area Styling */
        .gradio-container .upload-wrap {
            background: linear-gradient(135deg, rgba(30, 30, 50, 0.8) 0%, rgba(40, 40, 60, 0.8) 100%) !important;
            border: 3px dashed rgba(102, 126, 234, 0.5);
            border-radius: 20px;
            padding: 3rem;
            text-align: center;
            transition: all 0.3s ease;
            margin: 1rem 0;
            animation: bounceIn 1s ease-out 0.6s both;
        }

        @keyframes bounceIn {
            0% {
                opacity: 0;
                transform: scale(0.3);
            }
            50% {
                opacity: 1;
                transform: scale(1.05);
            }
            70% {
                transform: scale(0.9);
            }
            100% {
                opacity: 1;
                transform: scale(1);
            }
        }

        .gradio-container .upload-wrap:hover {
            border-color: #667eea;
            background: linear-gradient(135deg, rgba(40, 40, 60, 0.9) 0%, rgba(50, 50, 70, 0.9) 100%) !important;
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        }

        /* Accordion Styling */
        .gradio-container .accordion {
            background: rgba(30, 30, 50, 0.8) !important;
            border-radius: 16px;
            margin: 1rem 0;
            border: 1px solid rgba(102, 126, 234, 0.3);
            overflow: hidden;
            transition: all 0.3s ease;
            animation: slideInLeft 0.8s ease-out both;
        }

        .gradio-container .accordion:nth-child(odd) {
            animation-delay: 0.1s;
        }
        .gradio-container .accordion:nth-child(even) {
            animation-delay: 0.2s;
        }

        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        .gradio-container .accordion:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }

        /* Accordion Headers */
        .gradio-container .accordion .label-wrap {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            font-weight: 600;
            font-size: 1.1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .gradio-container .accordion .label-wrap::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }

        .gradio-container .accordion .label-wrap:hover::before {
            left: 100%;
        }

        .gradio-container .accordion .label-wrap:hover {
            background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
            transform: scale(1.02);
        }

        /* Accordion Content */
        .gradio-container .accordion .accordion-content {
            background: rgba(20, 20, 40, 0.9) !important;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            max-height: 0;
            opacity: 0;
            overflow: hidden;
        }

        .gradio-container .accordion.open .accordion-content {
            max-height: 2000px;
            opacity: 1;
            padding: 2rem;
        }

        /* Button Styling */
        .gradio-container .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 12px;
            padding: 1rem 2rem;
            color: white;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            margin: 0.5rem;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7);
            }
            70% {
                box-shadow: 0 0 0 10px rgba(102, 126, 234, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(102, 126, 234, 0);
            }
        }

        .gradio-container .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
        }

        .gradio-container .btn:active {
            transform: translateY(0);
        }

        /* Special Button Styles */
        .gradio-container .btn.primary {
            background: linear-gradient(135deg, #f1f1fa 0%, #f0faf0 100%);
        }

        .gradio-container .btn.danger {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        }

        .gradio-container .btn.secondary {
            background: linear-gradient(135deg, #f0f0f0 0%, #fafafa 100%);
        }

        /* Slider Styling */
        .gradio-container input[type="range"] {
            -webkit-appearance: none;
            appearance: none;
            height: 8px;
            border-radius: 5px;
            background: linear-gradient(to right, #667eea, #764ba2);
            outline: none;
            transition: all 0.3s ease;
        }

        .gradio-container input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
        }

        .gradio-container input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
        }

        /* Checkbox Styling */
        .gradio-container input[type="checkbox"] {
            width: 20px;
            height: 20px;
            accent-color: #667eea;
            border-radius: 4px;
            transition: all 0.3s ease;
        }

        .gradio-container input[type="checkbox"]:checked {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        /* Dropdown Styling */
        .gradio-container select {
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .gradio-container select:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            outline: none;
        }

        /* Text Input Styling */
        .gradio-container input[type="text"] {
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .gradio-container input[type="text"]:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            outline: none;
        }

        /* Image Display Styling */
        .gradio-container .image-container {
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            animation: zoomIn 1s ease-out 0.8s both;
            max-width: 450px;
        }

        @keyframes zoomIn {
            from {
                opacity: 0;
                transform: scale(0.8);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }

        .gradio-container .image-container:hover {
            transform: scale(1.02);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        }

        /* Fixed Image Container Dimensions */
        .gradio-container [data-testid="image"] {
            width: 100% !important;
            height: 400px !important;
            object-fit: contain !important;
            background: rgba(30, 30, 50, 0.8) !important;
            border-radius: 12px !important;
            border: 2px solid rgba(102, 126, 234, 0.5) !important;
            padding: 1rem !important;
        }

        /* Download File Widget Styling */
        .gradio-container .image-display-area [data-testid="file"] {
            margin-top: 1rem !important;
            background: rgba(30, 30, 50, 0.8) !important;
            border: 2px solid rgba(102, 126, 234, 0.3) !important;
            border-radius: 12px !important;
            padding: 1rem !important;
        }

        .gradio-container .image-display-area [data-testid="file"] label {
            color: #64748b !important;
            font-weight: 600 !important;
        }

        .gradio-container [data-testid="image"] img {
            width: 100% !important;
            height: 100% !important;
            object-fit: contain !important;
            border-radius: 8px !important;
        }

        /* Image Comparison Layout */
        .image-comparison {
            display: flex;
            gap: 1rem;
            align-items: flex-start;
            margin: 1rem 0;
        }

        .image-comparison .image-wrapper {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .image-comparison .image-wrapper h4 {
            margin: 0.5rem 0;
            color: #495057;
            font-size: 1rem;
            font-weight: 600;
        }

        .image-comparison .image-wrapper [data-testid="image"] {
            height: 350px !important;
        }

        /* Loading Animation */
        .gradio-container .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Responsive Design - Using Gradio's actual selectors */
        @media (max-width: 768px) {
            /* Mobile Layout */
            .gradio-container {
                padding: 0 !important;
            }

            /* Mobile Typography */
            h1, .gradio-container h1 {
                font-size: 1.8rem !important;
                margin: 1rem 0 !important;
            }

            /* Mobile Accordion */
            .gradio-container [data-testid="accordion"] {
                margin: 0.5rem 0 !important;
            }

            .gradio-container [data-testid="accordion"] > div {
                margin: 0.25rem 0 !important;
            }

            /* Mobile Rows - Force single column */
            .gradio-container [data-testid="row"] {
                flex-direction: column !important;
                gap: 0.5rem !important;
            }

            .gradio-container [data-testid="row"] > div {
                width: 100% !important;
                flex: none !important;
            }

            /* Mobile Buttons */
            .gradio-container button {
                width: 100% !important;
                margin: 0.25rem 0 !important;
                padding: 0.75rem 1rem !important;
                min-height: 44px !important;
            }

            /* Mobile Form Elements */
            .gradio-container input,
            .gradio-container select,
            .gradio-container textarea {
                width: 100% !important;
                margin: 0.25rem 0 !important;
            }

            /* Mobile Sliders */
            .gradio-container input[type="range"] {
                width: 100% !important;
            }

            /* Mobile Images */
            .gradio-container .image-container,
            .gradio-container [data-testid="image"] {
                width: 100% !important;
                max-width: 100% !important;
            }

            /* Mobile Download Widget */
            .gradio-container .image-display-area [data-testid="file"] {
                margin-top: 0.5rem !important;
                padding: 0.75rem !important;
            }
        }

        /* Tablet Responsive */
        @media (min-width: 769px) and (max-width: 1024px) {
            .gradio-container {
                padding: 0 !important;
            }

            .gradio-container [data-testid="row"] {
                display: grid !important;
                grid-template-columns: repeat(2, 1fr) !important;
                gap: 1rem !important;
            }

            .gradio-container button {
                padding: 0.75rem 1.5rem !important;
            }
        }
        /* Desktop Responsive */
        @media (min-width: 1025px) {
            .gradio-container {
                max-width: 1600px !important;
                margin: 0 auto !important;
                padding: 0 !important;
            }
            /* Grid layout for accordion container */
            .gradio-container .accordion-container {
                display: grid !important;
                grid-template-columns: repeat(2, 1fr) !important;
                gap: 1.5rem !important;
                margin-bottom: 2rem !important;
            }

            /* Grid layout for controls within accordions */
            .gradio-container .accordion-container [data-testid="row"] {
                display: grid !important;
                grid-template-columns: repeat(2, 1fr) !important;
                gap: 1rem !important;
                margin-bottom: 1rem !important;
            }

            /* Single column for some controls that need more space */
            .gradio-container .accordion-container [data-testid="row"].single-column {
                grid-template-columns: 1fr !important;
            }
        }

        /* Large Desktop Responsive */
        @media (min-width: 1400px) {
            .gradio-container {
                max-width: 1800px !important;
            }

            .gradio-container .accordion-container {
                grid-template-columns: repeat(3, 1fr) !important;
            }

            .gradio-container .accordion-container [data-testid="row"] {
                grid-template-columns: repeat(3, 1fr) !important;
            }
        }

        /* Ultra-wide Responsive */
        @media (min-width: 1920px) {
            .gradio-container {
                max-width: 2000px !important;
            }
            .gradio-container .accordion-container {
                grid-template-columns: repeat(4, 1fr) !important;}

            .gradio-container .accordion-container [data-testid="row"] {
                grid-template-columns: repeat(4, 1fr) !important;
            }
        }

        /* Force Dark Theme Throughout */
        .gradio-container {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
            color: #e2e8f0 !important;
        }

        .gradio-container * {
            color:#64748b !important;
        }

        .gradio-container input,
        .gradio-container select,
        .gradio-container textarea {
            background: rgba(30, 30, 50, 0.8) !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
            color: #ffffff !important;
        }

        .gradio-container label {
            color: #64748b !important;
        }

        /* Upload Button Styling */
        .gradio-container .upload-button {
            width: 200px !important;
            margin: 0 auto !important;
            display: block !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 1rem 2rem !important;
            color: white !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            position: relative !important;
            overflow: hidden !important;
            animation: subtlePulse 3s ease-in-out infinite !important;
        }

        .gradio-container .upload-button:hover {
            transform: translateY(-2px) scale(1.05) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
            animation: none !important;
        }

        .gradio-container .upload-button:active {
            transform: translateY(0) scale(1) !important;
        }

        @keyframes subtlePulse {
            0%, 100% {
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }
            50% {
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
            }
        }

        /* Upload Button Ripple Effect */
        .gradio-container .upload-button::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }

        .gradio-container .upload-button:active::before {
            width: 300px;
            height: 300px;
        }

        /* Print Responsive */
        @media print {
            .gradio-container {
                background: white !important;
                box-shadow: none !important;
            }

            .gradio-container .btn {
                display: none;
            }

            .gradio-container .accordion .label-wrap {
                background: #f8f9fa !important;
                color: #000 !important;
            }
        }

        /* Additional Responsive CSS for Gradio Components */
        @media (max-width: 768px) {
            /* Force mobile layout on all Gradio components */
            .gradio-container .form {
                display: flex !important;
                flex-direction: column !important;
                gap: 0.5rem !important;
            }

            .gradio-container .form > * {
                width: 100% !important;
                margin: 0.25rem 0 !important;
            }

            /* Mobile checkbox and radio groups */
            .gradio-container .checkbox,
            .gradio-container .radio {
                display: flex !important;
                flex-direction: column !important;
                gap: 0.5rem !important;
            }

            /* Mobile slider containers */
            .gradio-container .slider {
                width: 100% !important;
            }

            /* Mobile dropdown containers */
            .gradio-container .dropdown {
                width: 100% !important;
            }

            /* Mobile text input containers */
            .gradio-container .textbox {
                width: 100% !important;
            }

            /* Mobile color picker */
            .gradio-container .colorpicker {
                width: 100% !important;
            }

            /* Mobile image layout - stack vertically */
            .image-comparison {
                flex-direction: column !important;
                gap: 0.5rem !important;
            }
            .image-comparison > div {
                width: min(100%, 400px);
                margin: 0 auto;
            }

            .image-comparison .image-wrapper [data-testid="image"] {
                height: 300px !important;
            }
        }

        /* Tablet image layout */
        @media (min-width: 769px) and (max-width: 1024px) {
            .image-comparison {
                flex-direction: row !important;
                gap: 1rem !important;
            }

            .image-comparison .image-wrapper [data-testid="image"] {
                height: 320px !important;
            }
        }

        /* Responsive Utilities */
        .mobile-only {
            display: none;
        }

        .desktop-only {
            display: block;
        }

        @media (max-width: 768px) {
            .mobile-only {
                display: block;
            }

            .desktop-only {
                display: none;
            }
        }

        /* Touch-friendly improvements */
        @media (hover: none) and (pointer: coarse) {
            .gradio-container .btn {
                min-height: 44px;
                min-width: 44px;
            }

            .gradio-container input[type="range"]::-webkit-slider-thumb {
                width: 32px;
                height: 32px;
            }

            .gradio-container .accordion .label-wrap {
                min-height: 44px;
                display: flex;
                align-items: center;
            }
        }

        /* High DPI displays */
        @media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
            .gradio-container .image-container {
                image-rendering: -webkit-optimize-contrast;
                image-rendering: crisp-edges;
            }
        }

        /* Landscape orientation on mobile */
        @media (max-width: 768px) and (orientation: landscape) {
            .gradio-container {
                padding: 0.25rem;
            }

            .gradio-container h1 {
                font-size: 1.5rem;
                margin: 0.5rem 0;
            }

            .gradio-container .accordion .label-wrap {
                padding: 0.75rem;
            }

            .gradio-container .accordion .accordion-content {
                padding: 0.75rem;
            }
        }

        /* Floating Elements */
        .gradio-container::before {
            content: '';
            position: fixed;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
            animation: float 20s ease-in-out infinite;
            z-index: -1;
        }

        @keyframes float {
            0%, 100% {
                transform: translateY(0px) rotate(0deg);
            }
            50% {
                transform: translateY(-20px) rotate(180deg);
            }
        }

        /* Text Area Styling */
        .gradio-container textarea {
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            resize: vertical;
        }

        .gradio-container textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            outline: none;
        }

        /* Color Picker Styling */
        .gradio-container input[type="color"] {
            width: 50px;
            height: 50px;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        .gradio-container input[type="color"]:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
        }

        /* Additional Dark Theme Fixes */
        .gradio-container .form {
            background: rgba(20, 20, 40, 0.5) !important;
        }

        .gradio-container .form .form-group {
            background: transparent !important;
        }

        .gradio-container .form input,
        .gradio-container .form select,
        .gradio-container .form textarea {
            background: rgba(30, 30, 50, 0.8) !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
            color: #ffffff !important;
        }

        .gradio-container .form input:focus,
        .gradio-container .form select:focus,
        .gradio-container .form textarea:focus {
            background: rgba(40, 40, 60, 0.9) !important;
            border-color: #667eea !important;
        }

        .gradio-container .form label {
            color: #e2e8f0 !important;
        }

        .gradio-container .form .text-sm {
            color: #94a3b8 !important;
        }

        /* Comprehensive Upload Area Dark Theme */
        .gradio-container .upload-area,
        .gradio-container .upload-zone,
        .gradio-container .dropzone,
        .gradio-container .file-upload-area {
            background: rgba(30, 30, 50, 0.8) !important;
            border: 2px dashed rgba(102, 126, 234, 0.5) !important;
            color: #e2e8f0 !important;
        }

        .gradio-container .upload-area:hover,
        .gradio-container .upload-zone:hover,
        .gradio-container .dropzone:hover,
        .gradio-container .file-upload-area:hover {
            background: rgba(40, 40, 60, 0.9) !important;
            border-color: #667eea !important;
        }

        /* Upload text and labels */
        .gradio-container .upload-area *,
        .gradio-container .upload-zone *,
        .gradio-container .dropzone *,
        .gradio-container .file-upload-area * {
            color: #e2e8f0 !important;
        }

        /* Main Layout Styling */
        .gradio-container .main-layout {
            display: flex !important;
            gap: 2rem !important;
            align-items: flex-start !important;
        }

        .gradio-container .image-display-area {
            flex: 2 !important;
            min-width: 0 !important;
        }

        .gradio-container .features-panel {
            flex: 1 !important;
            max-width: 400px !important;
        }

        .gradio-container .scrollable-features {
            max-height: 80vh !important;
            overflow-y: auto !important;
            padding-right: 1rem !important;
            background: rgba(20, 20, 40, 0.5) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
        }

        .gradio-container .scrollable-features::-webkit-scrollbar {
            width: 8px !important;
        }

        .gradio-container .scrollable-features::-webkit-scrollbar-track {
            background: rgba(30, 30, 50, 0.5) !important;
            border-radius: 4px !important;
        }

        .gradio-container .scrollable-features::-webkit-scrollbar-thumb {
            background: rgba(102, 126, 234, 0.6) !important;
            border-radius: 4px !important;
        }

        .gradio-container .scrollable-features::-webkit-scrollbar-thumb:hover {
            background: rgba(102, 126, 234, 0.8) !important;
        }

        /* Mobile Layout */
        @media (max-width: 768px) {
            .gradio-container .main-layout {
                flex-direction: column !important;
                gap: 1rem !important;
            }

            .gradio-container .features-panel {
                max-width: 100% !important;
            }

            .gradio-container .scrollable-features {
                max-height: 60vh !important;
            }
        }

        /* Tab Styling */
        .gradio-container .main-tabs {
            background: rgba(30, 30, 50, 0.95) !important;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            overflow: hidden;
            margin: 2rem 0;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }

        .gradio-container .main-tabs .tab-nav {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-bottom: none;
            padding: 0;
        }

        .gradio-container .main-tabs .tab-nav button {
            background: transparent;
            border: none;
            color: rgba(255, 255, 255, 0.8);
            font-weight: 600;
            font-size: 1.1rem;
            padding: 1.5rem 2rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .gradio-container .main-tabs .tab-nav button:hover {
            color: white;
            background: rgba(255, 255, 255, 0.1);
            transform: translateY(-2px);
        }

        .gradio-container .main-tabs .tab-nav button.selected {
            color: white;
            background: rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .gradio-container .main-tabs .tab-nav button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }

        .gradio-container .main-tabs .tab-nav button:hover::before {
            left: 100%;
        }

        .gradio-container .main-tabs .tab-content {
            padding: 2rem;
            background: rgba(20, 20, 40, 0.9) !important;
            min-height: 600px;
        }

        /* Tab Content Animations */
        .gradio-container .tab-content {
            animation: fadeInUp 0.6s ease-out;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Mobile Tab Styling */
        @media (max-width: 768px) {
            .gradio-container .main-tabs .tab-nav {
                flex-direction: column;
            }

            .gradio-container .main-tabs .tab-nav button {
                width: 100%;
                padding: 1rem;
                font-size: 1rem;
            }

            .gradio-container .main-tabs .tab-content {
                padding: 1rem;
            }
        }

             /* RADIO BUTTON STYLING */
        /* Make radio buttons bigger and more visible */
        input[type="radio"] {
            accent-color: #667eea !important;
            width: 22px !important;
            height: 22px !important;
            cursor: pointer !important;
        }
        /* Style all radio button containers */
        .form input[type="radio"] {
            appearance: none !important;
            -webkit-appearance: none !important;
            width: 22px !important;
            height: 22px !important;
            border: 2px solid rgba(102, 126, 234, 0.5) !important;
            border-radius: 50% !important;
            cursor: pointer !important;
            background: transparent !important;
            position: relative !important;
            margin-right: 10px !important;
        }
        .form input[type="radio"]:hover {
            border-color: #667eea !important;
            box-shadow: 0 0 8px rgba(102, 126, 234, 0.3) !important;
        }
        .form input[type="radio"]:checked {
            background: #667eea !important;
            border-color: #667eea !important;
            box-shadow: 0 0 12px rgba(102, 126, 234, 0.6) !important;
        }
        .form input[type="radio"]:checked::after {
            content: '';
            position: absolute !important;
            top: 50% !important;
            left: 50% !important;
            width: 8px !important;
            height: 8px !important;
            background: white !important;
            border-radius: 50% !important;
            transform: translate(-50%, -50%) !important;
        }
        /* CHECKBOX STYLING */
        /* Make checkboxes bigger and more visible */
        input[type="checkbox"] {
            accent-color: #667eea !important;
            width: 22px !important;
            height: 22px !important;
            cursor: pointer !important;
        }
        .form input[type="checkbox"] {
            appearance: none !important;
            -webkit-appearance: none !important;
            width: 22px !important;
            height: 22px !important;
            border: 2px solid rgba(102, 126, 234, 0.5) !important;
            border-radius: 4px !important;
            cursor: pointer !important;
            background: transparent !important;
            position: relative !important;
            margin-right: 10px !important;
            transition: all 0.3s ease !important;
        }
        .form input[type="checkbox"]:hover {
            border-color: #667eea !important;
            box-shadow: 0 0 8px rgba(102, 126, 234, 0.3) !important;
        }
        .form input[type="checkbox"]:checked {
            background: #10b981 !important;
            border-color: #10b981 !important;
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.5) !important;
        }
        .form input[type="checkbox"]:checked::after {
            content: '✓';
            position: absolute !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            color: white !important;
            font-size: 14px !important;
            font-weight: bold !important;
            line-height: 1 !important;
        }
        /* Style the parent wrapper of radio/checkbox options */
        .form > div > div:has(input[type="radio"]),
        .form > div > div:has(input[type="checkbox"]) {
            padding: 0.75rem !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(102, 126, 234, 0.2) !important;
            margin-bottom: 0.5rem !important;
        }
        .form > div > div:has(input[type="radio"]:checked),
        .form > div > div:has(input[type="checkbox"]:checked) {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
            border-color: #667eea !important;
        }

        # end of CSS fo
        """
HEADER_CSS = """<div style="text-align: center; padding: 1rem 0;">
            <div style="display: inline-flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                <div style="font-size: 3rem; animation: bounce 2s infinite;"></div>
                <h1 style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                background-clip: text; font-size: 3.5rem; font-weight: 800;
                margin: 0; letter-spacing: -0.02em;">Imagaims</h1>
                <div style="font-size: 3rem; animation: bounce 2s infinite 0.5s;"></div>
            </div>
            <p style="color: #64748b; font-size: 1.2rem; max-width: 800px; margin: 0 auto; line-height: 1.6;">
                Transform your images with professional-grade filters, stunning effects, and creative tools.
                From vintage vibes to futuristic glitch art - your imagination is the only limit!
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem; flex-wrap: wrap;">
                <div style="background: rgba(102, 126, 234, 0.1); padding: 0.5rem 1rem;
                 border-radius: 20px; border: 1px solid rgba(102, 126, 234, 0.2);">
                    <span style="color: #667eea; font-weight: 600;">20+ Effects</span>
                </div>
                <div style="background: rgba(102, 126, 234, 0.1); padding: 0.5rem 1rem; border-radius: 20px;
                 border: 1px solid rgba(102, 126, 234, 0.2);">
                    <span style="color: #667eea; font-weight: 600;">Real-time Preview</span>
                </div>
                <div style="background: rgba(102, 126, 234, 0.1);
                padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid rgba(102, 126, 234, 0.2);">
                    <span style="color: #667eea; font-weight: 600;">Instant Export</span>
                </div>
            </div>
        </div>
        """
TAB_CSS = """
                <script>
                    document.addEventListener('DOMContentLoaded', function() {
                        // Accordion functionality with smooth animations
                        const accordions = document.querySelectorAll('.accordion');
                        accordions.forEach((accordion, index) => {
                            const header = accordion.querySelector('.label-wrap');
                            const content = accordion.querySelector('.accordion-content');

                            // Add staggered animation delay
                            accordion.style.animationDelay = `${index * 0.1}s`;

                            header.addEventListener('click', function() {
                                const isOpen = accordion.classList.contains('open');

                                // Close all other accordions
                                accordions.forEach(acc => {
                                    if (acc !== accordion) {
                                        acc.classList.remove('open');
                                        const otherContent = acc.querySelector('.accordion-content');
                                        otherContent.style.maxHeight = '0';
                                        otherContent.style.opacity = '0';
                                        otherContent.style.padding = '0 2rem';
                                    }
                                });

                                // Toggle current accordion
                                if (!isOpen) {
                                    accordion.classList.add('open');
                                    content.style.maxHeight = content.scrollHeight + 'px';
                                    content.style.opacity = '1';
                                    content.style.padding = '2rem';
                                } else {
                                    accordion.classList.remove('open');
                                    content.style.maxHeight = '0';
                                    content.style.opacity = '0';
                                    content.style.padding = '0 2rem';
                                }
                            });
                        });

                        // Add hover effects to buttons
                        document.querySelectorAll('button').forEach(button => {
                            button.addEventListener('mouseenter', function() {
                                this.style.transform = 'translateY(-3px) scale(1.05)';
                                this.style.boxShadow = '0 10px 30px rgba(102, 126, 234, 0.3)';
                            });

                            button.addEventListener('mouseleave', function() {
                                this.style.transform = 'translateY(0) scale(1)';
                                this.style.boxShadow = '0 4px 12px rgba(102, 126, 234, 0.2)';
                            });
                        });

                        // Add loading states to interactive elements
                        document.querySelectorAll('input, select, textarea').forEach(element => {
                            element.addEventListener('focus', function() {
                                this.style.transform = 'scale(1.02)';
                                this.style.boxShadow = '0 0 0 4px rgba(102, 126, 234, 0.1)';
                            });

                            element.addEventListener('blur', function() {
                                this.style.transform = 'scale(1)';
                                this.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.05)';
                            });
                        });

                        // Add ripple effect to buttons
                        document.querySelectorAll('button').forEach(button => {
                            button.addEventListener('click', function(e) {
                                const ripple = document.createElement('span');
                                const rect = this.getBoundingClientRect();
                                const size = Math.max(rect.width, rect.height);
                                const x = e.clientX - rect.left - size / 2;
                                const y = e.clientY - rect.top - size / 2;

                                ripple.style.cssText = `
                                    position: absolute;
                                    width: ${size}px;
                                    height: ${size}px;
                                    left: ${x}px;
                                    top: ${y}px;
                                    background: rgba(255, 255, 255, 0.3);
                                    border-radius: 50%;
                                    transform: scale(0);
                                    animation: ripple 0.6s ease-out;
                                    pointer-events: none;
                                `;

                                this.style.position = 'relative';
                                this.style.overflow = 'hidden';
                                this.appendChild(ripple);

                                setTimeout(() => ripple.remove(), 600);
                            });
                        });

                        // Add CSS for ripple animation
                        const style = document.createElement('style');
                        style.textContent = `
                            @keyframes ripple {
                                to {
                                    transform: scale(4);
                                    opacity: 0;
                                }
                            }
                            @keyframes bounce {
                                0%, 20%, 50%, 80%, 100% {
                                    transform: translateY(0);
                                }
                                40% {
                                    transform: translateY(-10px);
                                }
                                60% {
                                    transform: translateY(-5px);
                                }
                            }
                        `;
                        document.head.appendChild(style);
                    });
                </script>
                """
