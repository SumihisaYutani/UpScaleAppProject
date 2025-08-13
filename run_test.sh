#!/bin/bash
echo "🎬 UpScale App - Unix Environment Test"
echo ""

# Try different Python commands
echo "🔍 Testing Python availability..."

if command -v python3 &> /dev/null; then
    echo "✅ python3 command available"
    python3 quick_test.py
elif command -v python &> /dev/null; then
    echo "✅ python command available"
    python quick_test.py
else
    echo "❌ No Python installation found"
    echo ""
    echo "📝 Please install Python 3.8+ from your package manager:"
    echo "   Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "   macOS: brew install python3"
    echo "   Or download from https://python.org"
fi

echo ""
echo "Press Enter to continue..."
read