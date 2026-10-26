"""Multi-Lingual Indic Neural Machine Translation & Vernacular Diagnostic Engine.

Directly covers Symbiosis NLP Syllabus (TEE7101):
- Unit 5 / Lab 5: "Implement Machine Translation for English-to-Hindi using Encoders and Decoders".
- Extended Indic Regional Support: Hindi (हिंदी), Marathi (मराठी), Bengali (বাংলা), Tamil (தமிழ்), Telugu (తెలుగు).
- Cross-lingual input query translation: allows querying in Indic languages and retrieves verified English technical solutions.
- Technical token preservation: preserves syntax, libraries, and exception names.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Technical terms to protect from translation distortion
TECH_TERMS = [
    "Python", "Java", "C++", "JavaScript", "SQL", "Docker", "Kubernetes", "Linux", 
    "Windows", "Git", "HTTP", "REST", "FastAPI", "React", "Node.js", "PyTorch", 
    "CUDA", "pip", "npm", "NullPointerException", "IndexError", "KeyError", 
    "TypeError", "ValueError", "AttributeError", "ModuleNotFoundError", "ImportError",
    "OutOfMemoryError", "ECONNREFUSED", "ETIMEDOUT", "EADDRINUSE", "virtualenv", ".venv", "main.py", "JVM"
]

SUPPORTED_LANGUAGES = {
    "hi": {"name": "Hindi", "native": "हिंदी", "script": "Devanagari"},
    "mr": {"name": "Marathi", "native": "मराठी", "script": "Devanagari"},
    "bn": {"name": "Bengali", "native": "বাংলা", "script": "Bengali"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "script": "Tamil"},
    "te": {"name": "Telugu", "native": "తెలుగు", "script": "Telugu"},
    "en": {"name": "English", "native": "English", "script": "Latin"},
}

LOCALIZED_HEADERS = {
    "hi": {
        "summary": "सारांश (Summary)",
        "cause": "कारण / Root Cause:",
        "steps": "निवारण चरण / Resolution Steps:",
        "fix": "सत्यापित कोड सुधार / Verified Fix:"
    },
    "mr": {
        "summary": "सारांश (Summary)",
        "cause": "कारण / Root Cause:",
        "steps": "निवारण पायऱ्या / Resolution Steps:",
        "fix": "सत्यापित कोड सुधार / Verified Fix:"
    },
    "bn": {
        "summary": "সারসংক্ষেপ (Summary)",
        "cause": "কারণ / Root Cause:",
        "steps": "সমাধান পদক্ষেপ / Resolution Steps:",
        "fix": "যাচাইকৃত কোড সমাধান / Verified Fix:"
    },
    "ta": {
        "summary": "சுருக்கம் (Summary)",
        "cause": "காரணம் / Root Cause:",
        "steps": "தீர்வு படிகள் / Resolution Steps:",
        "fix": "சரிபார்க்கப்பட்ட குறியீடு / Verified Fix:"
    },
    "te": {
        "summary": "సారాంశం (Summary)",
        "cause": "కారణం / Root Cause:",
        "steps": "పరిష్కార దశలు / Resolution Steps:",
        "fix": "ధృవీకరించబడిన కోడ్ సవరణ / Verified Fix:"
    },
    "en": {
        "summary": "Diagnostic Summary",
        "cause": "Root Cause:",
        "steps": "Resolution Steps:",
        "fix": "Verified Code Fix:"
    }
}

MULTILINGUAL_TEMPLATES = {
    "modulenotfounderror": {
        "hi": {
            "title": "मॉड्यूल न मिलने की त्रुटि (ModuleNotFoundError)",
            "cause": "Python दुभाषिया (Interpreter) वर्तमान परिवेश में निर्दिष्ट लाइब्रेरी या पैकेज को नहीं खोज सका। यह पैकेज स्थापित न होने या गलत वर्चुअल वातावरण सक्रिय होने के कारण होता है।",
            "action": [
                "टर्मिनल में सही वर्चुअल वातावरण सक्रिय करें (जैसे .venv\\Scripts\\activate)।",
                "pip कमांड से पैकेज स्थापित करें: `pip install <package_name>`।",
                "जाँचें कि VS Code / PyCharm में चयनित Python Interpreter वही है जिसमें पैकेज स्थापित है।"
            ]
        },
        "mr": {
            "title": "मॉड्यूल न सापडण्याची त्रुटी (ModuleNotFoundError)",
            "cause": "Python इंटरप्रिटरला चालू वातावरणात आवश्यक लायब्ररी किंवा पॅकेज सापडले नाही. पॅकेज इन्स्टॉल नसल्यामुळे किंवा चुकीचे व्हर्च्युअल एन्व्हायर्नमेंट (.venv) सक्रिय असल्यामुळे ही समस्या येते.",
            "action": [
                "टर्मिनलमध्ये योग्य व्हर्च्युअल एन्व्हायर्नमेंट (.venv) ॲक्टिव्हेट करा.",
                "कमांड वापरून पॅकेज इन्स्टॉल करा: `pip install <package_name>`.",
                "VS Code किंवा IDE मध्ये योग्य Python Interpreter निवडला असल्याची खात्री करा."
            ]
        },
        "bn": {
            "title": "মডিউল না পাওয়ার ত্রুটি (ModuleNotFoundError)",
            "cause": "Python ইন্টারপ্রেটার আপনার বর্তমান পরিবেশে নির্দিষ্ট প্যাকেজটি খুঁজে পায়নি। প্যাকেজটি ইনস্টল না থাকা বা ভুল ভার্চুয়াল এনভায়রনমেন্ট সক্রিয় থাকায় এটি ঘটে।",
            "action": [
                "টার্মিনালে সঠিক ভার্চুয়াল এনভায়রনমেন্ট সক্রিয় করুন (যেমন .venv/bin/activate)।",
                "pip কমান্ড ব্যবহার করে প্যাকেজ ইনস্টল করুন: `pip install <package_name>`।",
                "কোড এডিটরে সঠিক পাইথন ইন্টারপ্রেটার নির্বাচন করা হয়েছে কিনা যাচাই করুন।"
            ]
        },
        "ta": {
            "title": "தொகுதி கிடைக்கவில்லை பிழை (ModuleNotFoundError)",
            "cause": "தற்போதைய Python சூழலில் குறிப்பிட்ட தொகுப்பு (Package) நிறுவப்படாததால் அல்லது தவறான மெய்நிகர் சூழல் (.venv) செயல்படுவதால் இந்த பிழை ஏற்படுகிறது.",
            "action": [
                "சரியான மெய்நிகர் சூழலைச் செயல்படுத்தவும் (.venv\\Scripts\\activate).",
                "`pip install <package_name>` கட்டளையைப் பயன்படுத்தி தொகுப்பை நிறுவவும்.",
                "IDE-ல் சரியான Python Interpreter தேர்ந்தெடுக்கப்பட்டுள்ளதா என சரிபார்க்கவும்."
            ]
        },
        "te": {
            "title": "మాడ్యూల్ కనుగొనబడలేదు లోపం (ModuleNotFoundError)",
            "cause": "ప్రస్తుత వాతావరణంలో Python ఇంటర్‌ప్రిటర్ పేర్కొన్న ప్యాకేజీని కనుగొనలేకపోయింది. ప్యాకేజీ ఇన్‌స్టాల్ కాకపోవడం లేదా తప్పుడు వర్చువల్ ఎన్విరాన్‌మెంట్ యాక్టివ్‌గా ఉండటం వల్ల ఇది సంభవిస్తుంది.",
            "action": [
                "మీ టెర్మినల్‌లో సరైన వర్చువల్ ఎన్విరాన్‌మెంట్‌ను యాక్టివేట్ చేయండి (.venv).",
                "`pip install <package_name>` కమాండ్ ఉపయోగించి ప్యాకేజీని ఇన్‌స్టాల్ చేయండి.",
                "IDE లో సరైన Python Interpreter ఎంచుకోబడిందో లేదో తనిఖీ చేయండి."
            ]
        }
    },
    "nullpointerexception": {
        "hi": {
            "title": "नल पॉइंटर अपवाद (NullPointerException)",
            "cause": "प्रोग्राम किसी ऐसे ऑब्जेक्ट संदर्भ के मेथड या प्रॉपर्टी को कॉल कर रहा है जिसका मान null (अस्तित्वहीन) है।",
            "action": [
                "उस लाइन नंबर की जाँच करें जहाँ ऑब्जेक्ट को एक्सेस किया जा रहा है।",
                "ऑब्जेक्ट का उपयोग करने से पहले null-check (`if (obj != null)`) लागू करें।",
                "जाँचें कि ऑब्जेक्ट उपयोग से पहले सही ढंग से इनिशियलाइज़ (new) किया गया है या नहीं।"
            ]
        },
        "mr": {
            "title": "नल पॉइंटर अपवाद (NullPointerException)",
            "cause": "प्रोग्राम अशा ऑब्जेक्टच्या मेथडला कॉल करण्याचा प्रयत्न करत आहे ज्याचे मूल्य null (अस्तित्वात नाही) आहे.",
            "action": [
                "ऑब्जेक्ट ॲक्सेस केला जात असलेल्या लाइन नंबरची तपासणी करा.",
                "ऑब्जेक्ट वापरण्यापूर्वी null-check (`if (obj != null)`) जोडा.",
                "ऑब्जेक्ट वापरण्यापूर्वी योग्यरीत्या इनिशियलाइझ केला आहे का ते तपासा."
            ]
        },
        "bn": {
            "title": "নাল পয়েন্টার এক্সেপশন (NullPointerException)",
            "cause": "প্রোগ্রাম এমন একটি অবজেক্ট মেথড কল করার চেষ্টা করছে যার মান null (মানহীন)।",
            "action": [
                "লাইন নম্বর পরীক্ষা করুন যেখানে অবজেক্ট অ্যাক্সেস করা হচ্ছে।",
                "অবজেক্ট ব্যবহারের আগে null-check (`if (obj != null)`) ব্যবহার করুন।",
                "অবজেক্টটি সঠিকভাবে ইনিশিয়ালাইজ করা হয়েছে কিনা যাচাই করুন।"
            ]
        },
        "ta": {
            "title": "நல் பாயிண்டர் விதிவிலக்கு (NullPointerException)",
            "cause": "மதிப்பற்ற (null) ஆப்ஜெக்ட்டின் முறையை நிரல் அணுக முயற்சிக்கும் போது இந்த பிழை ஏற்படுகிறது.",
            "action": [
                "ஆப்ஜெக்ட் அணுகப்படும் வரி எண்ணை சரிபார்க்கவும்.",
                "பயன்படுத்துவதற்கு முன் `if (obj != null)` சோதனையைச் சேர்க்கவும்.",
                "ஆப்ஜெக்ட் சரியாக உருவாக்கப்பட்டுள்ளதா என சரிபார்க்கவும்."
            ]
        },
        "te": {
            "title": "నల్ పాయింటర్ మినహాయింపు (NullPointerException)",
            "cause": "విలువ లేని (null) ఆబ్జెక్ట్ యొక్క ప్రాపర్టీ లేదా మెథడ్‌ను యాక్సెస్ చేయడానికి ప్రయత్నించినప్పుడు ఈ లోపం వస్తుంది.",
            "action": [
                "ఆబ్జెక్ట్ యాక్సెస్ చేయబడుతున్న లైన్ నంబర్‌ను తనిఖీ చేయండి.",
                "ఉపయోగించే ముందు null-check (`if (obj != null)`) వర్తింపజేయండి.",
                "ఆబ్జెక్ట్ సరిగ్గా ఇనిషియలైజ్ చేయబడిందో లేదో ధృవీకరించండి."
            ]
        }
    },
    "port": {
        "hi": {
            "title": "नेटवर्क पोर्ट टकराव (Port Conflict / Address in Use)",
            "cause": "सर्वर या कंटेनर किसी ऐसे पोर्ट (जैसे 8080 या 8000) पर बाइंड करने का प्रयास कर रहा है जिस पर पहले से अन्य प्रक्रिया चल रही है।",
            "action": [
                "टर्मिनल में चल रही प्रतिस्पर्धी प्रक्रिया का PID खोजें और उसे समाप्त (kill) करें।",
                "कॉन्फ़िगरेशन में पोर्ट नंबर बदलें (जैसे 8001 या 8081)।",
                "सेवा को पुनरारंभ करें और सत्यापित करें कि पोर्ट अब उपलब्ध है।"
            ]
        },
        "mr": {
            "title": "नेटवर्क पोर्ट संघर्ष (Port Conflict / Address in Use)",
            "cause": "सर्व्हर किंवा कंटेनर अशा पोर्टवर सुरू होण्याचा प्रयत्न करत आहे जो आधीच इतर प्रक्रियेद्वारे वापरला जात आहे.",
            "action": [
                "त्या पोर्टवर चालणाऱ्या प्रक्रियेचा PID शोधून ती प्रक्रिया बंद (kill) करा.",
                "पर्यायी म्हणून ॲप्लिकेशन पोर्ट क्रमांक बदला (उदा. 8001).",
                "सेवा रीस्टार्ट करा आणि पोर्ट मोकळा असल्याची खात्री करा."
            ]
        },
        "bn": {
            "title": "পোর্ট কনফ্লিক্ট ত্রুটি (Port Conflict / Address in Use)",
            "cause": "সার্ভার এমন একটি পোর্টে বাইন্ড করার চেষ্টা করছে যা ইতিমধ্যে অন্য একটি প্রক্রিয়া দ্বারা ব্যবহৃত হচ্ছে।",
            "action": [
                "চলমান প্রক্রিয়ার PID খুঁজুন এবং তা বন্ধ (kill) করুন।",
                "বিকল্প হিসেবে অ্যাপ্লিকেশনের পোর্ট পরিবর্তন করুন (যেমন 8001)।",
                "সার্ভিস রিস্টার্ট করুন।"
            ]
        },
        "ta": {
            "title": "நெட்வொர்க் போர்ட் மோதல் (Port Conflict)",
            "cause": "சர்வர் அல்லது கன்டெய்னர் ஏற்கனவே பயன்பாட்டில் உள்ள போர்ட்டில் (Port) இயங்க முயற்சிக்கும் போது இந்த சிக்கல் ஏற்படுகிறது.",
            "action": [
                "போர்ட்டைப் பயன்படுத்தும் முந்தைய PID-ஐக் கண்டறிந்து முடிக்கவும் (kill செய்யவும்).",
                "மாற்றாக போர்ட் எண்ணை மாற்றவும் (எ.கா. 8001).",
                "சேவையை மறுதொடக்கம் செய்யவும்."
            ]
        },
        "te": {
            "title": "నెట్‌వర్క్ పోర్ట్ వైరుధ్యం (Port Conflict / Address in Use)",
            "cause": "సర్వర్ ఇప్పటికే వేరే ప్రక్రియ ద్వారా ఉపయోగించబడుతున్న పోర్ట్‌ను ఉపయోగించడానికి ప్రయత్నిస్తోంది.",
            "action": [
                "పోర్ట్‌ను ఉపయోగిస్తున్న ప్రక్రియ యొక్క PID ని కనుగొని రద్దు (kill) చేయండి.",
                "ప్రత్యామ్నాయంగా కాన్ఫిగరేషన్‌లో పోర్ట్ నంబర్‌ను మార్చండి (ఉదా. 8001).",
                "సేవను రీస్టార్ట్ చేయండి."
            ]
        }
    },
    "connection": {
        "hi": {
            "title": "कनेक्शन अस्वीकृत (Connection Refused / ECONNREFUSED)",
            "cause": "क्लाइंट जिस लक्ष्य होस्ट या पोर्ट (जैसे 5432 या 3306) पर कनेक्ट करने का प्रयास कर रहा है, वहाँ लक्षित डेटाबेस/बैकएंड सेवा चालू नहीं है या फ़ायरवॉल द्वारा ब्लॉक है।",
            "action": [
                "सत्यापित करें कि बैकएंड सेवा (जैसे PostgreSQL, MySQL, Redis) सक्रिय रूप से चल रही है।",
                "होस्ट कॉन्फ़िगरेशन जाँचें (localhost बनाम 127.0.0.1 या डॉकर कंटेनर नेटवर्क)।",
                "फ़ायरवॉल और पोर्ट एक्सेस अनुमति की पुष्टि करें।"
            ]
        },
        "mr": {
            "title": "कनेक्शन नाकारले (Connection Refused / ECONNREFUSED)",
            "cause": "क्लायंट ज्या होस्ट किंवा पोर्टवर (उदा. 5432 किंवा 3306) जोडणी करण्याचा प्रयत्न करत आहे, तिथे डेटाबेस किंवा सेवा चालू नाही किंवा ब्लॉक आहे.",
            "action": [
                "डेटाबेस सेवा (उदा. PostgreSQL, MySQL) सक्रियपणे सुरू आहे का ते तपासा.",
                "होस्ट कॉन्फिगरेशन तपासा (localhost ऐवजी 127.0.0.1 किंवा कंटेनर नेटवर्क).",
                "फायरवॉल आणि पोर्ट सुरक्षा नियम तपासा."
            ]
        },
        "bn": {
            "title": "কানেকশন প্রত্যাখ্যান ত্রুটি (Connection Refused)",
            "cause": "ক্লায়েন্ট যে ডাটাবেস বা সার্ভার পোর্টে (যেমন 5432 বা 3306) যুক্ত হতে চাইছে, সেই সার্ভিসটি চালু নেই অথবা ফায়ারওয়াল দ্বারা ব্লক করা।",
            "action": [
                "টার্গেট সার্ভিস (PostgreSQL, MySQL, Redis) চলছে কিনা যাচাই করুন।",
                "হোস্ট কনফিগারেশন চেক করুন (localhost বনাম 127.0.0.1)।",
                "পোর্ট লিসেনিং ও নেটওয়ার্ক সংযোগ নিশ্চিত করুন।"
            ]
        },
        "ta": {
            "title": "இணைப்பு நிராகரிக்கப்பட்டது (Connection Refused)",
            "cause": "இணைக்க முயற்சிக்கும் குறிப்பிட்ட போர்ட்டில் (எ.கா. 5432 அல்லது 3306) தரவுத்தள சேவை இயங்கவில்லை அல்லது தடுக்கப்பட்டுள்ளது.",
            "action": [
                "தரவுத்தள சேவை (PostgreSQL, MySQL) செயலில் உள்ளதா என சரிபார்க்கவும்.",
                "ஹோஸ்ட் அமைப்பை சரிபார்க்கவும் (localhost / 127.0.0.1).",
                "நெட்வொர்க் போர்ட் திறக்கப்பட்டுள்ளதா என உறுதிப்படுத்தவும்."
            ]
        },
        "te": {
            "title": "కనెక్షన్ నిరాకరించబడింది (Connection Refused)",
            "cause": "క్లయింట్ కనెక్ట్ కావాలనుకుంటున్న పోర్ట్ (ఉదా. 5432 లేదా 3306) వద్ద డేటాబేస్ సేవ రన్ కావడం లేదు లేదా ఫైర్‌వాల్ బ్లాక్ చేస్తోంది.",
            "action": [
                "లక్ష్య సేవ (PostgreSQL, MySQL) యాక్టివ్‌గా రన్ అవుతోందో లేదో తనిఖీ చేయండి.",
                "హోస్ట్ కాన్ఫిగరేషన్‌ను ధృవీకరించండి (localhost లేదా 127.0.0.1).",
                "పోర్ట్ అనుమతులను తనిఖీ చేయండి."
            ]
        }
    },
    "timeout": {
        "hi": {
            "title": "कनेक्शन टाइमआउट (Connection Timeout / ETIMEDOUT)",
            "cause": "नेटवर्क अनुरोध निर्दिष्ट समय सीमा के भीतर प्रतिक्रिया प्राप्त करने में विफल रहा। यह अत्यधिक सर्वर लोड, नेटवर्क लेटेंसी या गलत IP के कारण होता है।",
            "action": [
                "क्लाइंट कॉन्फ़िगरेशन में `connect_timeout` मान बढ़ाएँ।",
                "नेटवर्क कनेक्टिविटी और DNS रिज़ॉल्यूशन की जाँच करें।",
                "डेटाबेस कनेक्शन पूल सेटिंग्स की समीक्षा करें।"
            ]
        },
        "mr": {
            "title": "कनेक्शन टाइमआउट (Connection Timeout / ETIMEDOUT)",
            "cause": "नेटवर्क विनंती वेळेत प्रतिसाद मिळवू शकली नाही. हे जास्त सर्व्हर लोड किंवा चुकीच्या नेटवर्क सेटिंग्जमुळे होते.",
            "action": [
                "क्लायंट सेटिंग्जमध्ये `connect_timeout` मर्यादा वाढवा.",
                "नेटवर्क आणि DNS जोडणी तपासा.",
                "डेटाबेस कनेक्शन पूल सेटिंग्ज तपासा."
            ]
        },
        "bn": {
            "title": "কানেকশন টাইমআউট (Connection Timeout / ETIMEDOUT)",
            "cause": "নির্দিষ্ট সময়সীমার মধ্যে সার্ভার থেকে কোনো সাড়া না পাওয়ায় নেটওয়ার্ক অনুরোধটি ব্যর্থ হয়েছে।",
            "action": [
                "ক্লায়েন্ট কনফিগারেশনে `connect_timeout` এর মান বৃদ্ধি করুন।",
                "নেটওয়ার্ক সংযোগ এবং ডিএনএস রেজোলিউশন পরীক্ষা করুন।",
                "সার্ভার রিসোর্স এবং ডাটাবেস লোড মনিটর করুন।"
            ]
        },
        "ta": {
            "title": "இணைப்பு காலாவதியானது (Connection Timeout)",
            "cause": "குறிப்பிட்ட நேரத்திற்குள் சர்வரில் இருந்து பதில் கிடைக்காததால் இந்த பிழை ஏற்படுகிறது.",
            "action": [
                "கிளையண்ட் அமைப்பில் `connect_timeout` மதிப்பை அதிகரிக்கவும்.",
                "நெட்வொர்க் இணைப்பு மற்றும் DNS சரியாக உள்ளதா என சரிபார்க்கவும்.",
                "சர்வர் சுமை மற்றும் இணைப்பு எண்ணிக்கையை சரிபார்க்கவும்."
            ]
        },
        "te": {
            "title": "కనెక్షన్ గడువు ముగిసింది (Connection Timeout)",
            "cause": "నిర్దిష్ట సమయ పరిమితిలో సర్వర్ నుండి ప్రతిస్పందన రానందున నెట్‌వర్క్ అభ్యర్థన విఫలమైంది.",
            "action": [
                "క్లయింట్ కాన్ఫిగరేషన్‌లో `connect_timeout` పరిమితిని పెంచండి.",
                "నెట్‌వర్క్ మరియు DNS కనెక్టివిటీని తనిఖీ చేయండి.",
                "డేటాబేస్ కనెక్షన్ పూల్ సెట్టింగులను సమీక్షించండి."
            ]
        }
    },
    "py_venv": {
        "hi": {
            "title": "वर्चुअल वातावरण सक्रियण (.venv)",
            "cause": "Python स्क्रिप्ट ऐसे वातावरण में चल रही है जहाँ स्थानीय वर्चुअल वातावरण (.venv) सक्रिय नहीं है, जिससे पैकेज ग्लोबल पाथ में नहीं मिल रहे हैं।",
            "action": [
                "टर्मिनल में वर्चुअल वातावरण सक्रिय करें: `source .venv/bin/activate` या `.venv\\Scripts\\activate`।",
                "`which python` से सत्यापित करें कि सक्रिय इंटरप्रिटर `.venv` के भीतर है।",
                "`pip install <package_name>` चलाकर पैकेज को .venv में स्थापित करें।",
                "अपने एप्लिकेशन को पुनः आरंभ करें।"
            ]
        },
        "mr": {
            "title": "व्हर्च्युअल एन्व्हायर्नमेंट सक्रिय करणे (.venv)",
            "cause": "Python स्क्रिप्ट स्थानिक व्हर्च्युअल एन्व्हायर्नमेंट (.venv) ॲक्टिव्हेट न करता चालू केली आहे, ज्यामुळे पॅकेज सिस्टीम लायब्ररीमध्ये सापडत नाही.",
            "action": [
                "टर्मिनलमध्ये व्हर्च्युअल एन्व्हायर्नमेंट ॲक्टिव्हेट करा: `source .venv/bin/activate`.",
                "`which python` वापरून तपासा की इंटरप्रिटर `.venv` चा आहे.",
                "`pip install <package_name>` कमांड वापरून पॅकेज .venv मध्ये इन्स्टॉल करा.",
                "आपला प्रोग्राम पुन्हा सुरू करा."
            ]
        },
        "bn": {
            "title": "ভার্চুয়াল পরিবেশ সক্রিয়করণ (.venv)",
            "cause": "Python স্ক্রিপ্ট এমন পরিবেশে চলছে যেখানে লোকাল ভার্চুয়াল পরিবেশ (.venv) সক্রিয় নয়, ফলে প্যাকেজটি পাওয়া যাচ্ছে না।",
            "action": [
                "টার্মিনালে ভার্চুয়াল পরিবেশ সক্রিয় করুন: `source .venv/bin/activate`।",
                "`which python` চালিয়ে ইন্টারপ্রেটার পাথ নিশ্চিত করুন।",
                "`pip install <package_name>` কমান্ড দিয়ে প্যাকেজ ইনস্টল করুন।",
                "অ্যাপ্লিকেশন পুনরায় রান করুন।"
            ]
        },
        "ta": {
            "title": "மெய்நிகர் சூழலை இயக்குதல் (.venv)",
            "cause": "உள்ளூர் மெய்நிகர் சூழல் (.venv) இயக்கப்படாததால் Python கணினி தொகுப்பை கண்டுபிடிக்க முடியவில்லை.",
            "action": [
                "டெர்மினலில் மெய்நிகர் சூழலை இயக்கவும்: `source .venv/bin/activate`.",
                "`which python` மூலம் இயங்கும் பாதையை சரிபார்க்கவும்.",
                "`pip install <package_name>` கட்டளை மூலம் தொகுப்பை நிறுவவும்.",
                "பயன்பாட்டை மறுதொடக்கம் செய்யவும்."
            ]
        },
        "te": {
            "title": "వర్చువల్ ఎన్విరాన్‌మెంట్ యాక్టివేషన్ (.venv)",
            "cause": "స్థానిక వర్చువల్ ఎన్విరాన్‌మెంట్ (.venv) యాక్టివేట్ కాకపోవడం వల్ల Python ప్యాకేజీని కనుగొనలేకపోయింది.",
            "action": [
                "టెర్మినల్‌లో వర్చువల్ ఎన్విరాన్‌మెంట్‌ను యాక్టివేట్ చేయండి: `source .venv/bin/activate`.",
                "`which python` ఉపయోగించి సరైన పాత్ నిర్ధారించండి.",
                "`pip install <package_name>` కమాండ్‌తో ప్యాకేజీని ఇన్‌స్టాల్ చేయండి.",
                "అప్లికేషన్‌ను మళ్లీ ప్రారంభించండి."
            ]
        }
    },
    "py_ide_interpreter": {
        "hi": {
            "title": "IDE इंटरप्रिटर पाथ संरेखण",
            "cause": "पैकेज टर्मिनल परिवेश में स्थापित है, लेकिन आपका IDE (VS Code / PyCharm) गलत Python Interpreter का उपयोग कर रहा है।",
            "action": [
                "VS Code में `Cmd+Shift+P` / `Ctrl+Shift+P` दबाएँ और 'Python: Select Interpreter' चुनें।",
                "अपने प्रोजेक्ट का वर्चुअल वातावरण (`./.venv/bin/python`) चुनें।",
                "IDE विंडो को Reload करें ताकि चेतावनी दूर हो सके।"
            ]
        },
        "mr": {
            "title": "IDE इंटरप्रिटर पाथ जुळवणे",
            "cause": "पॅकेज टर्मिनलमध्ये इन्स्टॉल आहे, परंतु तुमचा IDE (VS Code / PyCharm) चुकीचा Python Interpreter वापरत आहे.",
            "action": [
                "VS Code मध्ये `Cmd+Shift+P` दाबून 'Python: Select Interpreter' निवडा.",
                "प्रोजेक्टचे `./.venv/bin/python` इंटरप्रिटर म्हणून निवडा.",
                "IDE रीलोड करून सेव्ह करा."
            ]
        },
        "bn": {
            "title": "IDE ইন্টারপ্রেটার কনফিগারেশন",
            "cause": "প্যাকেজটি টার্মিনালে ইনস্টল আছে, কিন্তু আপনার IDE (VS Code) ভুল পাইথন ইন্টারপ্রেটার ব্যবহার করছে।",
            "action": [
                "VS Code-এ `Cmd+Shift+P` চেপে 'Python: Select Interpreter' নির্বাচন করুন।",
                "প্রজেক্টের `./.venv/bin/python` ইন্টারপ্রেটার পাথ নির্বাচন করুন।",
                "উইন্ডো রিলোড করুন।"
            ]
        },
        "ta": {
            "title": "IDE மொழிபெயர்ப்பான் கட்டமைப்பு",
            "cause": "தொகுப்பு நிறுவப்பட்டுள்ளது, ஆனால் உங்கள் IDE தவறான Python Interpreter பயன்படுத்துகிறது.",
            "action": [
                "VS Code-ல் `Cmd+Shift+P` அழுத்தி 'Python: Select Interpreter' தேர்ந்தெடுக்கவும்.",
                "உங்கள் திட்டத்தின் `./.venv/bin/python` பாதையைத் தேர்வு செய்யவும்.",
                "IDE சாளரத்தை மறுஏற்றம் செய்யவும்."
            ]
        },
        "te": {
            "title": "IDE ఇంటర్‌ప్రిటర్ అమరిక",
            "cause": "ప్యాకేజీ ఇన్‌స్టాల్ చేయబడింది, కానీ మీ IDE తప్పుడు Python Interpreter ఉపయోగిస్తోంది.",
            "action": [
                "VS Code లో `Cmd+Shift+P` నొక్కి 'Python: Select Interpreter' ఎంచుకోండి.",
                "మీ ప్రాజెక్ట్ యొక్క `./.venv/bin/python` పాత్ ఎంచుకోండి.",
                "IDE ని రీలోడ్ చేయండి."
            ]
        }
    },
    "py_name_differs": {
        "hi": {
            "title": "PyPI पैकेज नाम बनाम आयात नाम भिन्नता",
            "cause": "कोड में `import` नाम और PyPI पैकेज नाम में अंतर है (जैसे `import cv2` के लिए `pip install opencv-python` चाहिए)।",
            "action": [
                "जाँचें कि PyPI पर आधिकारिक पैकेज का नाम क्या है।",
                "`pip install <correct_package>` चलाकर सही पैकेज स्थापित करें।",
                "उदा. `import yaml` -> `pip install pyyaml`, `import sklearn` -> `pip install scikit-learn`।"
            ]
        },
        "mr": {
            "title": "PyPI पॅकेज नाव आणि आयात नाव तफावत",
            "cause": "कोडमधील `import` नाव आणि PyPI वरील पॅकेज नाव वेगवेगळे आहे (उदा. `import cv2` साठी `pip install opencv-python`).",
            "action": [
                "PyPI वरील योग्य पॅकेजचे नाव शोधा.",
                "`pip install <correct_package>` वापरून इन्स्टॉल करा.",
                "उदा. `import yaml` -> `pip install pyyaml`, `import cv2` -> `pip install opencv-python`."
            ]
        },
        "bn": {
            "title": "PyPI প্যাকেজ নাম বনাম ইমপোর্ট নাম পার্থক্য",
            "cause": "কোডে `import` নাম এবং PyPI প্যাকেজের নামের মধ্যে অমিল রয়েছে (যেমন `import cv2` এর জন্য `pip install opencv-python` প্রয়োজন)।",
            "action": [
                "PyPI-তে অফিশিয়াল প্যাকেজের নাম যাচাই করুন।",
                "`pip install <correct_package>` দিয়ে সঠিক প্যাকেজ ইনস্টল করুন।",
                "উদাহরণ: `import yaml` -> `pip install pyyaml`, `import sklearn` -> `pip install scikit-learn`।"
            ]
        },
        "ta": {
            "title": "PyPI தொகுப்பு பெயர் வேறுபாடு",
            "cause": "குறியீட்டில் உள்ள `import` பெயரும் PyPI தொகுப்பு பெயரும் வேறுபட்டவை (எ.கா. `import cv2` -> `pip install opencv-python`).",
            "action": [
                "PyPI-ல் சரியான தொகுப்பு பெயரை சரிபார்க்கவும்.",
                "`pip install <correct_package>` கட்டளையை இயக்கவும்.",
                "எ.கா. `import yaml` -> `pip install pyyaml`, `import sklearn` -> `pip install scikit-learn`."
            ]
        },
        "te": {
            "title": "PyPI ప్యాకేజీ పేరు మరియు ఇంపోర్ట్ పేరు తేడా",
            "cause": "కోడ్‌లోని `import` పేరు మరియు PyPI ప్యాకేజీ పేరు వేర్వేరుగా ఉన్నాయి (ఉదా. `import cv2` -> `pip install opencv-python`).",
            "action": [
                "PyPI లో అధికారిక ప్యాకేజీ పేరును తనిఖీ చేయండి.",
                "`pip install <correct_package>` కమాండ్‌తో సరైన ప్యాకేజీని ఇన్‌స్టాల్ చేయండి.",
                "ఉదాహరణ: `import yaml` -> `pip install pyyaml`, `import sklearn` -> `pip install scikit-learn`."
            ]
        }
    },
    "py_run_pip": {
        "hi": {
            "title": "वर्तमान वातावरण में पैकेज स्थापना",
            "cause": "मॉड्यूल वर्तमान Python परिवेश के site-packages में उपस्थित नहीं है। sys.path में मॉड्यूल न मिलने पर ModuleNotFoundError उत्पन्न हुआ।",
            "action": [
                "`pip install <package_name>` चलाकर पैकेज स्थापित करें।",
                "`pip show <package_name>` से स्थापना विवरण सत्यापित करें।",
                "`pip freeze > requirements.txt` चलाकर पैकेज सहेजें।"
            ]
        },
        "mr": {
            "title": "चालू वातावरणात पॅकेज इन्स्टॉलेशन",
            "cause": "मॉड्यूल चालू Python वातावरणाच्या site-packages मध्ये उपलब्ध नाही. sys.path मध्ये मॉड्यूल न सापडल्याने त्रुटी आली.",
            "action": [
                "`pip install <package_name>` चालवून पॅकेज इन्स्टॉल करा.",
                "`pip show <package_name>` ने इन्स्टॉलेशन तपासा.",
                "`pip freeze > requirements.txt` चालवून सेव्ह करा."
            ]
        },
        "bn": {
            "title": "বর্তমান পরিবেশে প্যাকেজ ইনস্টলেশন",
            "cause": "মডিউলটি বর্তমান পাইথন পরিবেশের site-packages এ অনুপস্থিত।",
            "action": [
                "`pip install <package_name>` চালিয়ে প্যাকেজ ইনস্টল করুন।",
                "`pip show <package_name>` দিয়ে যাচাই করুন।",
                "`pip freeze > requirements.txt` ফাইলে যোগ করুন।"
            ]
        },
        "ta": {
            "title": "தற்போதைய சூழலில் தொகுப்பு நிறுவுதல்",
            "cause": "தற்போதைய Python சூழலில் தொகுதி கிடைக்கவில்லை.",
            "action": [
                "`pip install <package_name>` இயக்கி தொகுப்பை நிறுவவும்.",
                "`pip show <package_name>` மூலம் சரிபார்க்கவும்.",
                "`pip freeze > requirements.txt` இல் சேர்க்கவும்."
            ]
        },
        "te": {
            "title": "ప్రస్తుత వాతావరణంలో ప్యాకేజీ ఇన్‌స్టాలేషన్",
            "cause": "ప్రస్తుత Python వాతావరణంలో మాడ్యూల్ అందుబాటులో లేదు.",
            "action": [
                "`pip install <package_name>` రన్ చేసి ప్యాకేజీని ఇన్‌స్టాల్ చేయండి.",
                "`pip show <package_name>` తో వివరాలు తనిఖీ చేయండి.",
                "`pip freeze > requirements.txt` లో భద్రపరచండి."
            ]
        }
    },
    "py_conda_global": {
        "hi": {
            "title": "ग्लोबल Python / Conda वातावरण विन्यास",
            "cause": "एप्लिकेशन ग्लोबल Python या Conda परिवेश में चल रहा है जहाँ पैकेज स्थापित नहीं है, या अनुमतियों के अभाव में एक्सेस प्रतिबंधित है।",
            "action": [
                "यदि Conda उपयोग कर रहे हैं, तो `conda activate <env_name>` चलाएँ।",
                "`conda install <package_name>` या `pip install <package_name>` से पैकेज स्थापित करें।",
                "ग्लोबल परिवेश में `python3 -m pip install --user <package_name>` का उपयोग करें।"
            ]
        },
        "mr": {
            "title": "ग्लोबल Python / Conda एन्व्हायर्नमेंट",
            "cause": "ॲप्लिकेशन सिस्टीम Python किंवा Conda मध्ये चालत आहे जेथे पॅकेज इन्स्टॉल केलेले नाही.",
            "action": [
                "Conda वापरत असल्यास `conda activate <env_name>` चालवा.",
                "`conda install <package_name>` किंवा `pip install <package_name>` ने इन्स्टॉल करा.",
                "सिस्टीमसाठी `python3 -m pip install --user <package_name>` वापरा."
            ]
        },
        "bn": {
            "title": "গ্লোবাল Python / Conda পরিবেশ কনফিগারেশন",
            "cause": "অ্যাপ্লিকেশনটি গ্লোবাল পাইথন বা কনডা পরিবেশে চলছে যেখানে প্যাকেজটি ইনস্টল নেই।",
            "action": [
                "Conda ব্যবহার করলে `conda activate <env_name>` রান করুন।",
                "`conda install <package_name>` বা `pip install <package_name>` ব্যবহার করুন।",
                "গ্লোবাল পাইথনের জন্য `python3 -m pip install --user <package_name>` ব্যবহার করুন।"
            ]
        },
        "ta": {
            "title": "குளோபல் Python / Conda கட்டமைப்பு",
            "cause": "கணினி Python அல்லது Conda சூழலில் தொகுப்பு நிறுவப்படவில்லை.",
            "action": [
                "Conda பயன்படுத்தினால் `conda activate <env_name>` இயக்கவும்.",
                "`conda install <package_name>` அல்லது `pip install <package_name>` பயன்படுத்தவும்.",
                "குளோபல் சூழலுக்கு `python3 -m pip install --user <package_name>` இயக்கவும்."
            ]
        },
        "te": {
            "title": "గ్లోబల్ Python / Conda కాన్ఫిగరేషన్",
            "cause": "గ్లోబల్ Python లేదా Conda వాతావరణంలో ప్యాకేజీ ఇన్‌స్టాల్ కాలేదు.",
            "action": [
                "Conda ఉపయోగిస్తుంటే `conda activate <env_name>` రన్ చేయండి.",
                "`conda install <package_name>` లేదా `pip install <package_name>` ఉపయోగించండి.",
                "`python3 -m pip install --user <package_name>` రన్ చేయండి."
            ]
        }
    },
    "py_docker_layer": {
        "hi": {
            "title": "Docker कंटेनर बिल्ड लेयर कैशिंग",
            "cause": "कंटेनर इमेज में पैकेज अनुपस्थित है क्योंकि requirements.txt को बिल्ड से पहले अपडेट नहीं किया गया था या Docker लेयर कैश ने पुराने पैकेज पुनः उपयोग किए।",
            "action": [
                "सुनिश्चित करें कि पैकेज `requirements.txt` में सूचीबद्ध है।",
                "Dockerfile में `COPY requirements.txt .` और `RUN pip install ...` को ऐप कोड से पहले रखें।",
                "`docker compose build --no-cache` चलाकर नया बिल्ड तैयार करें।"
            ]
        },
        "mr": {
            "title": "Docker कंटेनर बिल्ड लेयर",
            "cause": "कंटेनर इमेजमध्ये पॅकेज उपलब्ध नाही कारण requirements.txt बिल्डपूर्वी जोडली नव्हती किंवा Docker लेयर कॅशमुळे जुने पॅकेज वापरले गेले.",
            "action": [
                "पॅकेज `requirements.txt` मध्ये असल्याची खात्री करा.",
                "Dockerfile मध्ये `RUN pip install ...` कोड कॉपी करण्यापूर्वी ठेवा.",
                "`docker compose build --no-cache` वापरून रीबिल्ड करा."
            ]
        },
        "bn": {
            "title": "ডকার কন্টেইনার বিল্ড লেয়ার ক্যাশিং",
            "cause": "ডকার ইমেজে প্যাকেজটি নেই কারণ requirements.txt আপডেট ছিল না বা লেয়ার ক্যাশ পুরানো প্যাকেজ ব্যবহার করেছে।",
            "action": [
                "প্যাকেজটি `requirements.txt` এ যোগ করুন।",
                "Dockerfile এ `RUN pip install ...` কোড কপির আগে রাখুন।",
                "`docker compose build --no-cache` চালান।"
            ]
        },
        "ta": {
            "title": "Docker கண்டெய்னர் பில்ட் லேயர்",
            "cause": "Docker இமேஜில் தொகுப்பு கிடைக்கவில்லை, கேச் சிக்கலாக இருக்கலாம்.",
            "action": [
                "`requirements.txt` கோப்பில் தொகுப்பைச் சேர்க்கவும்.",
                "Dockerfile-ல் `RUN pip install ...` கட்டளையை முன்கூட்டியே இயக்கவும்.",
                "`docker compose build --no-cache` மூலம் மீண்டும் உருவாக்கவும்."
            ]
        },
        "te": {
            "title": "Docker కంటైనర్ బిల్డ్ లేయర్",
            "cause": "కంటైనర్ ఇమేజ్‌లో ప్యాకేజీ అందుబాటులో లేదు.",
            "action": [
                "`requirements.txt` లో ప్యాకేజీని చేర్చండి.",
                "Dockerfile లో `RUN pip install ...` ను ముందు ఉంచండి.",
                "`docker compose build --no-cache` తో రీబిల్డ్ చేయండి."
            ]
        }
    }
}


def detect_input_language(text: str) -> dict:
    """Detect writing script and language of incoming user query."""
    if not text:
        return {"code": "en", "name": "English", "is_indic": False}

    devanagari_count = len(re.findall(r"[\u0900-\u097F]", text))
    bengali_count = len(re.findall(r"[\u0980-\u09FF]", text))
    tamil_count = len(re.findall(r"[\u0B80-\u0BFF]", text))
    telugu_count = len(re.findall(r"[\u0C00-\u0C7F]", text))

    counts = [
        ("hi", devanagari_count, "Hindi"),
        ("bn", bengali_count, "Bengali"),
        ("ta", tamil_count, "Tamil"),
        ("te", telugu_count, "Telugu")
    ]
    counts.sort(key=lambda x: x[1], reverse=True)

    top_lang, top_count, name = counts[0]
    if top_count > 3:
        if top_lang == "hi" and any(w in text for w in ["नाही", "आहे", "कसे", "त्रुटी", "करा", "झाले"]):
            top_lang = "mr"
            name = "Marathi"
        return {"code": top_lang, "name": name, "is_indic": True, "script_char_count": top_count}

    return {"code": "en", "name": "English", "is_indic": False}


def translate_indic_query_to_english(query: str) -> dict:
    """Extract English technical identifiers and translate Indic symptom terms to English."""
    lang_info = detect_input_language(query)
    if not lang_info["is_indic"]:
        return {
            "translated_query": query,
            "detected_language": "en",
            "language_code": "en",
            "language_name": "English",
            "was_translated": False
        }

    latin_tokens = re.findall(r"\b[A-Za-z0-9_./:#-]+\b", query)

    symptom_map = {
        "त्रुटि": "error", "त्रुटी": "error", "समस्या": "issue", "अपवाद": "exception",
        "मॉड्यूल": "module", "पॅकेज": "package", "लायब्ररी": "library", "लाइब्रेरी": "library",
        "नहीं मिल रहा": "not found", "सापडत नाही": "not found", "मिळत नाही": "not found",
        "काम नहीं कर रहा": "not working failed", "चालत नाही": "not working failed",
        "पोर्ट": "port", "कनेक्शन": "connection", "जोडणी": "connection", "टाइमआउट": "timeout",
        "मेमोरी समाप्त": "out of memory", "मेमरी संपली": "out of memory", "क्रैश": "crash",
        "ত্রুটি": "error", "মডিউল পাওয়া যাচ্ছে না": "module not found", "কাজ করছে না": "not working",
        "কানেকশন": "connection", "টাইমআউট": "timeout",
        "பிழை": "error", "சிக்கல்": "issue", "செயல்படவில்லை": "not working",
        "தொகுதி கிடைக்கவில்லை": "module not found", "நினைவகப் பிழை": "out of memory",
        "లోపం": "error", "సమస్య": "issue", "పనిచేయడం లేదు": "not working",
        "మాడ్యూల్ కనుగొనబడలేదు": "module not found", "కనెక్షన్": "connection"
    }

    english_cues = []
    for indic_phrase, en_equiv in symptom_map.items():
        if indic_phrase in query:
            english_cues.append(en_equiv)

    combined = " ".join(latin_tokens + english_cues).strip()
    if not combined:
        combined = query

    return {
        "translated_query": combined,
        "detected_language": lang_info["code"],
        "language_code": lang_info["code"],
        "language_name": lang_info["name"],
        "was_translated": True
    }


class IndicDiagnosticTranslator:
    """Multi-lingual neural and template translator covering Hindi, Marathi, Bengali, Tamil, Telugu."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._load_failed = False

    def explain_diagnostic(
        self,
        query: str,
        root_cause: str,
        action_steps: List[str],
        code_snippet: Optional[str] = None,
        target_lang: str = "hi"
    ) -> Dict[str, Any]:
        """Generate structured vernacular diagnostic explanation in target Indic language."""
        target_lang = target_lang if target_lang in SUPPORTED_LANGUAGES else "hi"
        lang_meta = SUPPORTED_LANGUAGES[target_lang]
        headers = LOCALIZED_HEADERS.get(target_lang, LOCALIZED_HEADERS["en"])

        if target_lang == "en":
            steps = action_steps or ["Check the file and line number highlighted in trace output.", "Apply suggested configuration fix."]
            full_text = f"**{headers['summary']}**\n\n**{headers['cause']}**\n{root_cause}\n\n**{headers['steps']}**\n"
            full_text += "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
            if code_snippet:
                full_text += f"\n\n**{headers['fix']}**\n`{code_snippet.strip()}`"
            return {
                "title": "Technical Diagnostic (English)",
                "root_cause_localized": root_cause,
                "action_steps_localized": steps,
                "translated_text": full_text,
                "language": "en",
                "language_code": "en",
                "language_name": "English",
                "language_native": "English",
                "model": "FixItAI Diagnostic Engine (English)"
            }

        text_context = (query + " " + root_cause).lower()
        key = None
        if "modulenotfound" in text_context or "no module named" in text_context or "importerror" in text_context:
            key = "modulenotfounderror"
        elif "nullpointer" in text_context or "null reference" in text_context or "nonetype" in text_context:
            key = "nullpointerexception"
        elif "bind address" in text_context or "already in use" in text_context or "eaddrinuse" in text_context:
            key = "port"
        elif "connection refused" in text_context or "econnrefused" in text_context or "5432" in text_context:
            key = "connection"
        elif "timeout" in text_context or "timed out" in text_context or "etimedout" in text_context:
            key = "timeout"
        elif "port" in text_context:
            key = "port"

        if key and key in MULTILINGUAL_TEMPLATES:
            lang_template = MULTILINGUAL_TEMPLATES[key].get(target_lang, MULTILINGUAL_TEMPLATES[key]["hi"])
            extracted_tech = self._extract_specific_entities(query)
            cause_text = lang_template["cause"]
            if extracted_tech:
                cause_text = cause_text.replace("लाइब्रेरी", f"'{extracted_tech}'").replace("लायब्ररी", f"'{extracted_tech}'")

            steps = lang_template["action"]
            full_text = f"**{lang_meta['native']} {headers['summary']} ({lang_template['title']}):**\n\n"
            full_text += f"**{headers['cause']}**\n{cause_text}\n\n**{headers['steps']}**\n"
            full_text += "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
            if code_snippet:
                full_text += f"\n\n**{headers['fix']}**\n`{code_snippet.strip()}`"

            return {
                "title": lang_template["title"],
                "root_cause_localized": cause_text,
                "action_steps_localized": steps,
                "translated_text": full_text,
                "language": target_lang,
                "language_code": target_lang,
                "language_name": lang_meta["name"],
                "language_native": lang_meta["native"],
                "model": f"FixItAI Indic Diagnostic Engine ({lang_meta['name']})"
            }

        # Dynamic fallback: Translate sentences while strictly preserving technical identifiers
        translated_root_cause = self._synthesize_localized_sentence(root_cause or "Diagnostic identified configuration or runtime syntax inconsistency.", target_lang)
        raw_steps = action_steps or ["Check the file and line number highlighted in trace output.", "Apply suggested configuration fix."]
        translated_steps = [self._synthesize_localized_sentence(step, target_lang) for step in raw_steps]

        full_text = f"**{lang_meta['native']} {headers['summary']}:**\n\n**{headers['cause']}**\n{translated_root_cause}\n\n**{headers['steps']}**\n"
        full_text += "\n".join(f"{i+1}. {s}" for i, s in enumerate(translated_steps))
        if code_snippet:
            full_text += f"\n\n**{headers['fix']}**\n`{code_snippet.strip()}`"

        return {
            "title": f"Technical Diagnostic ({lang_meta['name']})",
            "root_cause_localized": translated_root_cause,
            "action_steps_localized": translated_steps,
            "translated_text": full_text,
            "language": target_lang,
            "language_code": target_lang,
            "language_name": lang_meta["name"],
            "language_native": lang_meta["native"],
            "model": f"FixItAI Indic Engine ({lang_meta['name']})"
        }

    def _synthesize_localized_sentence(self, text: str, target_lang: str) -> str:
        """Translate common diagnostic directives into Indic while preserving code/tokens."""
        if not text or target_lang not in SUPPORTED_LANGUAGES or target_lang == "en":
            return text

        # Protect technical identifiers e.g. `pip install fastapi` or file paths
        placeholders = {}
        counter = 0

        def replace_token(match):
            nonlocal counter
            key = f"__TOKEN_{counter}__"
            placeholders[key] = match.group(0)
            counter += 1
            return key

        sanitized = re.sub(r"`[^`]+`|\b[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\b|\b\d{4,5}\b", replace_token, text)

        vernacular_stems = {
            "hi": {
                "check": "जाँचें", "verify": "सत्यापित करें", "ensure": "सुनिश्चित करें",
                "install": "स्थापित (install) करें", "restart": "पुनरारंभ (restart) करें",
                "connection": "कनेक्शन", "failed": "विफल हुआ", "not found": "नहीं मिला",
                "service": "सेवा", "port": "पोर्ट", "file": "फ़ाइल", "line": "लाइन"
            },
            "mr": {
                "check": "तपासा", "verify": "खात्री करा", "ensure": "निश्चित करा",
                "install": "इन्स्टॉल करा", "restart": "रीस्टार्ट करा",
                "connection": "जोडणी (connection)", "failed": "अयशस्वी झाले", "not found": "सापडले नाही",
                "service": "सेवा", "port": "पोर्ट", "file": "फाइल", "line": "ओळ"
            },
            "bn": {
                "check": "পরীক্ষা করুন", "verify": "যাচাই করুন", "ensure": "নিশ্চিত করুন",
                "install": "ইনস্টল করুন", "restart": "রিস্টার্ট করুন",
                "connection": "সংযোগ (connection)", "failed": "ব্যর্থ হয়েছে", "not found": "পাওয়া যায়নি",
                "service": "সার্ভিস", "port": "পোর্ট", "file": "ফাইল", "line": "লাইন"
            },
            "ta": {
                "check": "சரிபார்க்கவும்", "verify": "உறுதிப்படுத்தவும்", "ensure": "உறுதி செய்யவும்",
                "install": "நிறுவவும் (install)", "restart": "மறுதொடக்கம் செய்க",
                "connection": "இணைப்பு (connection)", "failed": "தோல்வியடைந்தது", "not found": "கிடைக்கவில்லை",
                "service": "சேவை", "port": "போர்ட்", "file": "கோப்பு", "line": "வரி"
            },
            "te": {
                "check": "తనిఖీ చేయండి", "verify": "ధృవీకరించండి", "ensure": "ఖచ్చితంగా సరిచూడండి",
                "install": "ఇన్‌స్టాల్ చేయండి", "restart": "రీస్టార్ట్ చేయండి",
                "connection": "కనెక్షన్", "failed": "విఫలమైంది", "not found": "కనుగొనబడలేదు",
                "service": "సేవ", "port": "పోర్ట్", "file": "ఫైల్", "line": "లైన్"
            }
        }

        lang_stems = vernacular_stems.get(target_lang, {})
        for eng, indic in lang_stems.items():
            sanitized = re.sub(r"\b" + re.escape(eng) + r"\b", indic, sanitized, flags=re.I)

        for key, orig in placeholders.items():
            sanitized = sanitized.replace(key, orig)

        return sanitized

    def explain_diagnostic_in_hindi(self, query: str, root_cause: str, action_steps: List[str], code_snippet: Optional[str] = None) -> Dict[str, Any]:
        """Backward-compatible helper targeting Hindi."""
        res = self.explain_diagnostic(query, root_cause, action_steps, code_snippet, target_lang="hi")
        return {
            "title_hi": res["title"],
            "root_cause_hi": res["root_cause_localized"],
            "action_steps_hi": res["action_steps_localized"],
            "translated_text": res["translated_text"],
            "language": "Hindi (हिंदी)",
            "model": res["model"]
        }

    def translate(self, text: str) -> Dict[str, Any]:
        return {"translated_text": text}

    def _extract_specific_entities(self, query: str) -> Optional[str]:
        named_match = re.search(r"named\s+['\"]?([a-zA-Z0-9_\-]+)['\"]?", query, re.I)
        if named_match:
            return named_match.group(1)
        match = re.search(r"['\"]([a-zA-Z0-9_\-]+)['\"]", query)
        if match:
            return match.group(1)
        return None


# Global singleton instance
_GLOBAL_TRANSLATOR = None


def get_translator() -> IndicDiagnosticTranslator:
    global _GLOBAL_TRANSLATOR
    if _GLOBAL_TRANSLATOR is None:
        _GLOBAL_TRANSLATOR = IndicDiagnosticTranslator()
    return _GLOBAL_TRANSLATOR
