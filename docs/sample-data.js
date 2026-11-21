// Illustrative records transcribed from the five synthetic envelopes in sample_data/.
// These are static examples; no OCR model runs in GitHub Pages.
const POSTVISION_SAMPLES = {
    1: {
        recipient_name: "Mrs. Priya Smith",
        premise_building: "Plot No. 45, Lotus Tower",
        street_locality: "Gandhi Marg, Banjara Hills",
        city: "Pune",
        state: "Maharashtra",
        postal_code: "411001"
    },
    2: {
        recipient_name: "Neha Williams",
        premise_building: "Apt 4B, Shanti Niwas",
        street_locality: "Commercial Street, Jayanagar 4th Block",
        city: "Bengaluru",
        state: "Karnataka",
        postal_code: "560001"
    },
    3: {
        recipient_name: "Mr. John Miller",
        premise_building: "Apt 4B, Royal Palms",
        street_locality: "Church Street, Alwarpet",
        city: "Mumbai",
        state: "Maharashtra",
        postal_code: "400053"
    },
    4: {
        recipient_name: "Dr. Arjun Nair",
        premise_building: "Flat No. 304, Orchid Residency",
        street_locality: "Hill Road, Alwarpet",
        city: "Mumbai",
        state: "Maharashtra",
        postal_code: "400053"
    },
    5: {
        recipient_name: "Mrs. Olivia Jones",
        premise_building: "Flat No. 304, Gokul Vihar",
        street_locality: "Gandhi Marg, Sector 14",
        city: "Lucknow",
        state: "Uttar Pradesh",
        postal_code: "226001"
    }
};

function postVisionSampleImage(sampleId) {
    return `assets/sample_envelope_${sampleId}.png`;
}

function postVisionSampleResult(sampleId) {
    const address = POSTVISION_SAMPLES[sampleId];
    if (!address) throw new Error(`Sample #${sampleId} is unavailable.`);

    const lines = [
        ["From:", "return_address"],
        ["Dept of Logistics, Hub 4", "return_address"],
        ["Central Postal Terminal", "return_address"],
        [address.recipient_name, "destination_address"],
        [address.premise_building, "destination_address"],
        [address.street_locality, "destination_address"],
        [`${address.city}, ${address.state} ${address.postal_code}`, "destination_address"]
    ];
    const image = postVisionSampleImage(sampleId);
    return {
        success: true,
        sample_mode: true,
        timestamp: new Date().toISOString(),
        file_id: `sample_${sampleId}`,
        original_filename: `sample_envelope_${sampleId}.png`,
        original_image: image,
        annotated_image: image,
        num_lines_detected: lines.length,
        line_details: lines.map(([text, blockType], index) => ({
            line_number: index + 1,
            block_type: blockType,
            transcribed_text: text,
            confidence: null
        })),
        structured_address: { ...address, postal_validation: null, overall_confidence: null },
        dual_zone: {
            recipient: { ...address },
            sender: {
                detected: true,
                raw_text: "Dept of Logistics, Hub 4, Central Postal Terminal",
                parsed: { recipient_name: "Dept of Logistics", premise_building: "Hub 4", street_locality: "Central Postal Terminal" }
            }
        },
        vision_stages: {},
        vision_diagnostics: {},
        postage_stamp: { stamp_detected: true, cancellation_type: "Printed sample" }
    };
}
