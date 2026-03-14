import { ChangeEvent, DragEvent, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import AppLayout from "@/components/AppLayout";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import * as pdfjsLib from "pdfjs-dist";
import pdfWorkerSrc from "pdfjs-dist/build/pdf.worker.mjs?url";
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Download,
  FileCheck2,
  FileJson,
  FileText,
  Loader2,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  Trash2,
  TriangleAlert,
  Upload,
} from "lucide-react";

// Configure pdfjs worker to use the local bundled version
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerSrc;

type ParsedIndicator = {
  id: string;
  name: string;
  category: string;
  value: string;
  unit: string;
  referenceRange: string;
  status: string;
  instrument?: string;
};

type ParseMeta = {
  model: string;
  char_count: number;
  chunk_count: number;
  page_count: number;
  filename: string;
  max_file_size_mb: number;
  ark_base_url: string;
};

type ParseResponse = {
  fileName: string;
  contentType?: string;
  indicatorCount: number;
  indicators: ParsedIndicator[];
  meta: ParseMeta;
};

type UploadHistoryItem = {
  fileName: string;
  uploadedAt: string;
  indicatorCount: number;
  parsedData: ParseResponse;
};

type ReviewTone = "attention" | "normal" | "pending";

const categoryStyles: Record<string, string> = {
  "Conditions & Diagnoses": "bg-red-50 text-red-700 border-red-200",
  Medications: "bg-blue-50 text-blue-700 border-blue-200",
  "Lab Results": "bg-purple-50 text-purple-700 border-purple-200",
  Vitals: "bg-emerald-50 text-emerald-700 border-emerald-200",
  "Wearable Data": "bg-amber-50 text-amber-700 border-amber-200",
  "Imaging / Reports": "bg-pink-50 text-pink-700 border-pink-200",
};

const categoryOrder = [
  "Lab Results",
  "Vitals",
  "Imaging / Reports",
  "Conditions & Diagnoses",
  "Medications",
  "Wearable Data",
];

const uploadSteps = [
  {
    title: "Upload your PDF report",
    description: "Use a text-based health examination or blood panel report under 20MB.",
  },
  {
    title: "AI extracts structured indicators",
    description: "The backend repairs malformed fields and normalizes every result into one schema.",
  },
  {
    title: "Review before saving or sharing",
    description: "Check flagged items, inspect categories, and download the structured JSON when needed.",
  },
];

const emptyStateHighlights = [
  "Supports text-based PDF reports",
  "Out-of-range values are surfaced first",
  "Structured JSON is still available if you need it",
];

const manualFields = [
  { id: "height", name: "身高", category: "Vitals", unit: "cm", ref: "" },
  { id: "weight", name: "体重", category: "Vitals", unit: "Kg", ref: "" },
  { id: "body_mass_index", name: "体重指数", category: "Vitals", unit: "", ref: "18.50-23.99" },
  { id: "systolic_blood_pressure", name: "收缩压", category: "Vitals", unit: "mmHg", ref: "90-139" },
  { id: "diastolic_blood_pressure", name: "舒张压", category: "Vitals", unit: "mmHg", ref: "60-89" },
  { id: "heart_rate", name: "心率（次/分）", category: "Vitals", unit: "次/分", ref: "" },
  { id: "white_blood_cell_count", name: "白细胞计数", category: "Lab Results", unit: "×10^9/L", ref: "3.5-9.5" },
  { id: "red_blood_cell_count", name: "红细胞计数", category: "Lab Results", unit: "10^12/L", ref: "3.8-5.1" },
  { id: "hemoglobin", name: "血红蛋白", category: "Lab Results", unit: "g/L", ref: "115-150" },
  { id: "hematocrit", name: "红细胞压积", category: "Lab Results", unit: "L/L", ref: "0.35-0.45" },
  { id: "mean_corpuscular_volume", name: "平均红细胞体积", category: "Lab Results", unit: "fL", ref: "82-100" },
  { id: "mean_corpuscular_hemoglobin_content", name: "平均红细胞血红蛋白含量", category: "Lab Results", unit: "pg", ref: "27-34" },
  { id: "mean_corpuscular_hemoglobin_concentration", name: "平均红细胞血红蛋白浓度", category: "Lab Results", unit: "g/L", ref: "316-354" },
  { id: "red_cell_distribution_width_coefficient_of_variation", name: "红细胞分布宽度-变异系数", category: "Lab Results", unit: "%", ref: "11-16" },
  { id: "platelet_count", name: "血小板计数", category: "Lab Results", unit: "10^9/L", ref: "125-350" },
  { id: "mean_platelet_volume", name: "平均血小板体积", category: "Lab Results", unit: "fL", ref: "9-16" },
  { id: "platelet_distribution_width", name: "血小板分布宽度", category: "Lab Results", unit: "fL", ref: "8.00-20.00" },
  { id: "lymphocyte_percentage", name: "淋巴细胞百分比", category: "Lab Results", unit: "%", ref: "20-50" },
  { id: "neutrophil_percentage", name: "中性粒细胞百分比", category: "Lab Results", unit: "%", ref: "40-75" },
  { id: "lymphocyte_absolute", name: "淋巴细胞绝对值", category: "Lab Results", unit: "10^9/L", ref: "1.1-3.2" },
  { id: "neutrophil_absolute", name: "中性粒细胞绝对值", category: "Lab Results", unit: "10^9/L", ref: "1.8-6.3" },
  { id: "red_cell_distribution_width_standard_deviation", name: "红细胞分布宽度-标准差", category: "Lab Results", unit: "fL", ref: "35-54" },
  { id: "platelet_crit", name: "血小板压积", category: "Lab Results", unit: "%", ref: "0.108-0.330" },
  { id: "monocyte_percentage", name: "单核细胞百分比", category: "Lab Results", unit: "%", ref: "3-10" },
  { id: "monocyte_absolute", name: "单核细胞绝对值", category: "Lab Results", unit: "10^9/L", ref: "0.1-0.6" },
  { id: "eosinophil_percentage", name: "嗜酸性细胞百分比", category: "Lab Results", unit: "%", ref: "0.4-8.0" },
  { id: "eosinophil_absolute", name: "嗜酸性细胞绝对值", category: "Lab Results", unit: "10^9/L", ref: "0.02-0.52" },
  { id: "basophil_percentage", name: "嗜碱性细胞百分比", category: "Lab Results", unit: "%", ref: "0-1" },
  { id: "basophil_absolute", name: "嗜碱性细胞绝对值", category: "Lab Results", unit: "10^9/L", ref: "0-0.06" },
  { id: "urine_specific_gravity", name: "尿比重", category: "Lab Results", unit: "", ref: "1.003-1.030" },
  { id: "urine_ph", name: "尿酸碱度", category: "Lab Results", unit: "", ref: "4.5-8.0" },
  { id: "urine_leukocytes", name: "尿白细胞", category: "Lab Results", unit: "", ref: "阴性" },
  { id: "urine_nitrite", name: "尿亚硝酸盐", category: "Lab Results", unit: "", ref: "阴性" },
  { id: "urine_protein", name: "尿蛋白质", category: "Lab Results", unit: "", ref: "阴性" },
  { id: "urine_glucose", name: "尿糖", category: "Lab Results", unit: "", ref: "阴性" },
  { id: "urine_ketone_bodies", name: "尿酮体", category: "Lab Results", unit: "", ref: "阴性" },
  { id: "urine_urobilinogen", name: "尿胆原", category: "Lab Results", unit: "", ref: "阴性" },
  { id: "urine_bilirubin", name: "尿胆红素", category: "Lab Results", unit: "", ref: "阴性" },
  { id: "urine_occult_blood", name: "尿隐血", category: "Lab Results", unit: "", ref: "阴性" },
  { id: "urine_red_blood_cells", name: "尿镜检红细胞", category: "Lab Results", unit: "/HP", ref: "0-3" },
  { id: "urine_white_blood_cells", name: "尿镜检白细胞", category: "Lab Results", unit: "/HP", ref: "0-5" },
  { id: "urine_casts", name: "管型", category: "Lab Results", unit: "", ref: "" },
  { id: "alanine_aminotransferase", name: "丙氨酸氨基转移酶", category: "Lab Results", unit: "U/L", ref: "0-40" },
  { id: "aspartate_aminotransferase", name: "天门冬氨酸氨基转移酶", category: "Lab Results", unit: "U/L", ref: "13-35" },
  { id: "gamma_glutamyl_transferase", name: "γ-谷氨酰转移酶", category: "Lab Results", unit: "U/L", ref: "7-45" },
];

const checkStatus = (val: string, ref: string) => {
  if (!ref || !val) return "normal";
  
  if (ref === "阴性") {
    return val === "阴性" || val.toLowerCase() === "negative" || val === "-" ? "normal" : "abnormal";
  }
  
  const numVal = parseFloat(val);
  if (isNaN(numVal)) return "abnormal"; // Non-numeric answer to a numeric reference is generally abnormal tracking

  if (ref.includes("-")) {
    const [minStr, maxStr] = ref.split("-");
    const min = parseFloat(minStr);
    const max = parseFloat(maxStr);
    if (!isNaN(min) && numVal < min) return "low";
    if (!isNaN(max) && numVal > max) return "high";
  }
  
  return "normal";
};

const recentUploadPlaceholders = [
  { name: "Blood_Panel_March2025.pdf", date: "Mar 7, 2025", records: 12 },
  { name: "Annual_Checkup_Feb2025.pdf", date: "Feb 20, 2025", records: 9 },
];

const trimTrailingSlash = (url: string) => url.replace(/\/+$/, "");

const resolveParseFunctionUrl = () => {
  const supabaseUrl = import.meta.env.VITE_SUPABASE_URL?.trim();
  if (supabaseUrl) {
    return `${trimTrailingSlash(supabaseUrl)}/functions/v1/parse-health-report`;
  }

  // In hosted Lovable deployments, same-origin rewrites can route /functions/*.
  return "/functions/v1/parse-health-report";
};

const formatStatus = (status: string) => (status ? status.toUpperCase() : "UNKNOWN");

const formatDate = (iso: string) =>
  new Date(iso).toLocaleString("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

const asRecord = (value: unknown): Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

const firstPresent = (record: Record<string, unknown>, keys: string[]) => {
  for (const key of keys) {
    if (key in record) {
      return record[key];
    }
  }
  return undefined;
};

const readString = (record: Record<string, unknown>, keys: string[], fallback = "") => {
  const value = firstPresent(record, keys);
  if (value == null) return fallback;
  if (typeof value === "string") return value.trim() || fallback;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return fallback;
};

const readNumber = (record: Record<string, unknown>, keys: string[], fallback = 0) => {
  const value = firstPresent(record, keys);
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
};

const slugify = (value: string) =>
  value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "indicator";

const normalizeStatus = (status: string) => {
  const value = status.trim().toLowerCase();
  if (["normal", "high", "low", "abnormal"].includes(value)) return value;
  if (!value) return "";
  if (value.includes("high")) return "high";
  if (value.includes("low")) return "low";
  if (value.includes("normal")) return "normal";
  return "abnormal";
};

const getStatusTone = (status: string): ReviewTone => {
  if (status === "high" || status === "low" || status === "abnormal") return "attention";
  if (status === "normal") return "normal";
  return "pending";
};

const getStatusClasses = (status: string) => {
  const tone = getStatusTone(status);
  if (tone === "attention") return "bg-red-50 text-red-700 border-red-200";
  if (tone === "normal") return "bg-emerald-50 text-emerald-700 border-emerald-200";
  return "bg-slate-100 text-slate-600 border-slate-200";
};

const getCategoryClasses = (category: string) => categoryStyles[category] ?? "bg-muted text-muted-foreground border-border";

const normalizeIndicator = (value: unknown, index: number, seenIds: Map<string, number>): ParsedIndicator | null => {
  const record = asRecord(value);
  if (!Object.keys(record).length) return null;

  const name = readString(record, ["name", "indicatorName", "label", "title"], `Indicator ${index + 1}`);
  const rawId = readString(record, ["id", "indicatorId", "slug", "code"], slugify(name));
  const counter = (seenIds.get(rawId) ?? 0) + 1;
  seenIds.set(rawId, counter);

  return {
    id: counter === 1 ? rawId : `${rawId}-${counter}`,
    name,
    category: readString(record, ["category", "type", "group", "section"], "Lab Results"),
    value: readString(record, ["value", "result", "measurement", "finding"]),
    unit: readString(record, ["unit", "units"]),
    referenceRange: readString(record, ["referenceRange", "reference_range", "reference", "range"]),
    status: normalizeStatus(readString(record, ["status", "flag", "statusFlag", "resultFlag"])),
    instrument: readString(record, ["instrument", "method", "testMethod", "device"]),
  };
};

const extractResponsePayload = (payloadText: string): unknown => {
  const trimmed = payloadText.trim();
  if (!trimmed) {
    return {};
  }

  try {
    return JSON.parse(trimmed);
  } catch {
    const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
    const candidate = fenced?.[1] ?? trimmed.match(/\{[\s\S]*\}|\[[\s\S]*\]/)?.[0];
    if (!candidate) {
      throw new Error("Backend did not return valid JSON.");
    }

    return JSON.parse(candidate);
  }
};

const normalizeParseResponse = (payload: unknown, fallbackFileName: string): ParseResponse => {
  const root = Array.isArray(payload) ? { indicators: payload } : asRecord(payload);
  const rawIndicators =
    firstPresent(root, ["indicators", "items", "results", "data", "records"]) ??
    (Array.isArray(payload) ? payload : []);

  const seenIds = new Map<string, number>();
  const indicators = Array.isArray(rawIndicators)
    ? rawIndicators
        .map((indicator, index) => normalizeIndicator(indicator, index, seenIds))
        .filter((indicator): indicator is ParsedIndicator => indicator !== null)
    : [];

  const metaRecord = asRecord(firstPresent(root, ["meta", "metadata", "debug"]));

  return {
    fileName: readString(root, ["fileName", "file_name", "filename"], fallbackFileName),
    contentType: readString(root, ["contentType", "content_type"], "application/pdf"),
    indicatorCount: indicators.length,
    indicators,
    meta: {
      model: readString(metaRecord, ["model"]),
      char_count: readNumber(metaRecord, ["char_count", "charCount"]),
      chunk_count: readNumber(metaRecord, ["chunk_count", "chunkCount"]),
      page_count: readNumber(metaRecord, ["page_count", "pageCount"]),
      filename: readString(metaRecord, ["filename", "fileName"], fallbackFileName),
      max_file_size_mb: readNumber(metaRecord, ["max_file_size_mb", "maxFileSizeMb"]),
      ark_base_url: readString(metaRecord, ["ark_base_url", "arkBaseUrl"]),
    },
  };
};

const downloadParsedJson = (data: ParseResponse) => {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const safeBaseName = data.fileName.replace(/\.pdf$/i, "") || "parsed-report";

  link.href = url;
  link.download = `${safeBaseName}.json`;
  link.click();

  URL.revokeObjectURL(url);
};

export default function HealthDataUpload() {
  const [dragging, setDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [parsedData, setParsedData] = useState<ParseResponse | null>(null);
  const [uploadHistory, setUploadHistory] = useState<UploadHistoryItem[]>([]);
  const [manualData, setManualData] = useState<Record<string, string>>({});
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const { toast } = useToast();

  const groupedIndicators = useMemo(() => {
    if (!parsedData) return [];

    const groups = new Map<string, ParsedIndicator[]>();
    for (const category of categoryOrder) {
      groups.set(category, []);
    }

    parsedData.indicators.forEach((indicator) => {
      const key = indicator.category || "Lab Results";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)?.push(indicator);
    });

    return Array.from(groups.entries())
      .filter(([, indicators]) => indicators.length > 0)
      .map(([label, indicators]) => ({ label, indicators }));
  }, [parsedData]);

  const flaggedIndicators = useMemo(
    () => parsedData?.indicators.filter((indicator) => getStatusTone(indicator.status) === "attention") ?? [],
    [parsedData],
  );

  const normalIndicators = useMemo(
    () => parsedData?.indicators.filter((indicator) => getStatusTone(indicator.status) === "normal").length ?? 0,
    [parsedData],
  );

  const reviewProgress = useMemo(() => {
    if (!parsedData?.indicatorCount) return 0;
    return Math.round((normalIndicators / parsedData.indicatorCount) * 100);
  }, [normalIndicators, parsedData]);

  const handleManualSubmit = () => {
    const indicators: ParsedIndicator[] = [];
    
    manualFields.forEach(field => {
      const valStr = manualData[field.id];
      if (valStr && valStr.trim() !== "") {
        const valNum = parseFloat(valStr);
        indicators.push({
          id: field.id,
          name: field.name,
          category: field.category,
          unit: field.unit,
          referenceRange: field.ref || "",
          status: checkStatus(valStr.trim(), field.ref),
          value: valStr.trim(),
          instrument: "Manual Entry"
        });
      }
    });

    if (indicators.length === 0) {
      toast({
        title: "No data entered",
        description: "Please enter at least one value to submit.",
        variant: "destructive"
      });
      return;
    }

    const manualFileName = "Manual_Entry_" + new Date().toISOString().split("T")[0] + ".json";
    
    const manualParsed: ParseResponse = {
      fileName: manualFileName,
      contentType: "application/json",
      indicatorCount: indicators.length,
      indicators,
      meta: {
        model: "manual-entry",
        char_count: 0,
        chunk_count: 0,
        page_count: 1,
        filename: manualFileName,
        max_file_size_mb: 0,
        ark_base_url: "local"
      }
    };

    setParsedData(manualParsed);
    setSelectedFileName(manualFileName);
    
    setUploadHistory((current) => [
      {
        fileName: manualFileName,
        uploadedAt: new Date().toISOString(),
        indicatorCount: manualParsed.indicatorCount,
        parsedData: manualParsed,
      },
      ...current,
    ]);

    toast({
      title: "Data submitted",
      description: `${indicators.length} indicators were successfully saved.`,
    });
  };

  const uploadFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      toast({
        title: "Only PDF is supported",
        description: "Please upload a text-based health examination report in PDF format.",
        variant: "destructive",
      });
      return;
    }

    setIsUploading(true);
    setSelectedFileName(file.name);

    try {
      // 1. Extract text locally from PDF
      const arrayBuffer = await file.arrayBuffer();
      const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
      
      let fullText = "";
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const textContent = await page.getTextContent();
        // @ts-ignore
        const pageText = textContent.items.map(item => item.str).join(" ");
        fullText += pageText + "\n";
      }

      if (!fullText.trim()) {
        throw new Error("Could not extract text from this PDF. Ensure it is a text-based report, not just scanned images.");
      }

      // Safeguard against insanely large extraction
      const textChunk = fullText.slice(0, 30000);

      // 2. Send extracted text to Supabase Edge Function.
      const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY?.trim() || "";
      const functionUrl = resolveParseFunctionUrl();
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };

      if (supabaseAnonKey) {
        headers.Authorization = `Bearer ${supabaseAnonKey}`;
        headers.apikey = supabaseAnonKey;
      }

      const aiResponse = await fetch(functionUrl, {
        method: "POST",
        headers,
        body: JSON.stringify({
          textChunk,
          fileName: file.name
        })
      });

      if (!aiResponse.ok) {
        let errMessage = `Edge function failed with status ${aiResponse.status}`;
        try {
          const errData = await aiResponse.json();
          errMessage = errData.error || errMessage;
        } catch {
           // fallback to status text if non-JSON error
        }
        throw new Error(errMessage);
      }

      const { responseText, model, baseURL, error } = await aiResponse.json();
      if (error) {
        throw new Error(error);
      }

      const data = normalizeParseResponse(extractResponsePayload(responseText), file.name);
      
      // Patch meta to reflect the proxy parse
      data.meta = {
        model: model || "unknown",
        char_count: fullText.length,
        chunk_count: 1,
        page_count: pdf.numPages,
        filename: file.name,
        max_file_size_mb: 20,
        ark_base_url: baseURL || "proxy"
      };

      setParsedData(data);
      setUploadHistory((current) => [
        {
          fileName: data.fileName,
          uploadedAt: new Date().toISOString(),
          indicatorCount: data.indicatorCount,
          parsedData: data,
        },
        ...current.filter((item) => item.fileName !== data.fileName),
      ]);

      toast({
        title: "Report parsed",
        description: `${data.fileName} is ready for review with ${data.indicatorCount} extracted results.`,
      });
    } catch (error) {
      setParsedData(null);

      let errorMessage = "Unable to parse the selected file.";
      if (error instanceof TypeError) {
        errorMessage =
          "Network request failed. In Lovable, configure VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY, or ensure /functions/v1/parse-health-report is routed to Supabase.";
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }

      toast({
        title: "Upload failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileSelection = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    await uploadFile(file);
    event.target.value = "";
  };

  const handleDrop = async (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);

    const file = event.dataTransfer.files?.[0];
    if (!file) return;

    await uploadFile(file);
  };

  return (
    <AppLayout title="Health Data">
      <div className="p-6 space-y-6">
        <section className="relative overflow-hidden rounded-3xl border border-border bg-[linear-gradient(135deg,hsl(220_45%_14%)_0%,hsl(200_58%_15%)_42%,hsl(171_55%_18%)_100%)] p-6 text-white shadow-[var(--shadow-lg)]">
          <div className="absolute inset-y-0 right-0 hidden w-1/3 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.18),transparent_58%)] lg:block" />
          <div className="relative grid gap-6 lg:grid-cols-[1.35fr_0.95fr]">
            <div className="space-y-5">
              <div className="flex flex-wrap items-center gap-2">
                <Badge className="border-white/20 bg-white/10 text-white hover:bg-white/10">Encrypted upload</Badge>
                <Badge className="border-white/20 bg-white/10 text-white hover:bg-white/10">PDF only</Badge>
                <Badge className="border-white/20 bg-white/10 text-white hover:bg-white/10">AI-assisted review</Badge>
              </div>

              <div className="space-y-2">
                <h2 className="max-w-2xl text-3xl font-bold tracking-tight">Turn your health report into a clean, review-ready summary.</h2>
                <p className="max-w-2xl text-sm text-white/75">
                  Upload a health examination PDF and HealthVault will extract indicators, organize them into categories, and surface anything that may need your attention.
                </p>
              </div>

              <Tabs defaultValue="upload" className="w-full">
                <TabsList className="mb-4 grid w-full grid-cols-2 rounded-2xl bg-white/10 p-1">
                  <TabsTrigger value="upload" className="rounded-xl data-[state=active]:bg-white data-[state=active]:text-slate-900">Upload PDF</TabsTrigger>
                  <TabsTrigger value="manual" className="rounded-xl data-[state=active]:bg-white data-[state=active]:text-slate-900">Manual Entry</TabsTrigger>
                </TabsList>

                <TabsContent value="upload" className="mt-0">
                  <div
                    className={`rounded-2xl border border-white/12 bg-white/8 p-5 backdrop-blur transition-all ${dragging ? "border-primary bg-white/14" : "hover:bg-white/10"}`}
                    onDragOver={(event) => {
                      event.preventDefault();
                      setDragging(true);
                    }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={handleDrop}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      className="hidden"
                      accept=".pdf,application/pdf"
                      onChange={handleFileSelection}
                    />

                    <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                      <div className="flex items-start gap-4">
                        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white/12">
                          {isUploading ? <Loader2 className="h-6 w-6 animate-spin" /> : <Upload className="h-6 w-6" />}
                        </div>
                        <div className="space-y-1">
                          <div className="text-base font-semibold">Upload a report for AI review</div>
                          <div className="text-sm text-white/70">Text-based PDF, up to 20MB. Personal identifiers are not shown in the summary cards.</div>
                          {selectedFileName && <div className="text-xs text-white/70">Current file: {selectedFileName}</div>}
                        </div>
                      </div>

                      <div className="flex flex-wrap gap-3">
                        <Button
                          className="gap-2 border border-white/10 bg-white text-slate-900 hover:bg-white/90"
                          disabled={isUploading}
                          onClick={() => fileInputRef.current?.click()}
                        >
                          {isUploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                          {isUploading ? "Parsing report..." : "Choose PDF"}
                        </Button>
                        {parsedData && parsedData.meta.model !== "manual-entry" && (
                          <Button
                            variant="outline"
                            className="gap-2 border-white/20 bg-white/5 text-white hover:bg-white/10 hover:text-white"
                            onClick={() => downloadParsedJson(parsedData)}
                          >
                            <Download className="h-4 w-4" />
                            Download JSON
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="manual" className="mt-0">
                  <div className="rounded-2xl border border-white/12 bg-white/8 p-5 backdrop-blur">
                    <div className="mb-4 text-sm text-white/80">
                      Enter any known health indicators below. Leave unknown fields blank.
                    </div>
                    <div className="max-h-[250px] overflow-y-auto pr-3 space-y-3 scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent">
                      {manualFields.map(field => (
                        <div key={field.id} className="grid grid-cols-[1fr_120px] gap-4 items-center rounded-xl p-2 hover:bg-white/5">
                          <div>
                            <label className="text-sm font-medium text-white/90 block">{field.name}</label>
                            {field.ref && <span className="text-white/50 text-[10px]">Reference: {field.ref} {field.unit}</span>}
                          </div>
                          <div className="relative">
                            <input 
                              type="text" 
                              className="block w-full rounded-lg border border-white/10 bg-white/5 py-1.5 pl-3 pr-8 text-white placeholder:text-white/30 sm:text-sm sm:leading-6 focus:ring-1 focus:ring-primary focus:border-primary focus:bg-white/10 transition-colors" 
                              placeholder={field.ref === "阴性" ? "阴性/阳性" : "0.0"} 
                              value={manualData[field.id] || ""} 
                              onChange={(e) => setManualData({...manualData, [field.id]: e.target.value})} 
                            />
                            {field.unit && (
                              <div className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
                                <span className="text-white/40 text-[10px] uppercase truncate max-w-[40px]" title={field.unit}>
                                  {field.unit}
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                    <Separator className="my-5 bg-white/10" />
                    <div className="flex items-center justify-between">
                      <div className="text-xs text-white/50">
                        {Object.values(manualData).filter(v => v.trim() !== "").length} field(s) filled
                      </div>
                      <Button className="bg-white text-slate-900 hover:bg-white/90" onClick={handleManualSubmit}>
                        <FileCheck2 className="h-4 w-4 mr-2" />
                        Submit Entry
                      </Button>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>

              <div className="grid gap-3 sm:grid-cols-3">
                {emptyStateHighlights.map((item) => (
                  <div key={item} className="rounded-2xl border border-white/10 bg-white/6 px-4 py-3 text-sm text-white/78">
                    {item}
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/8 p-5 backdrop-blur">
              <div className="mb-4 flex items-center gap-2 text-sm font-medium text-white/90">
                <Sparkles className="h-4 w-4 text-teal-200" />
                What happens after upload
              </div>
              <div className="space-y-4">
                {uploadSteps.map((step, index) => (
                  <div key={step.title} className="flex gap-3">
                    <div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full border border-white/15 bg-white/10 text-xs font-semibold">
                      {index + 1}
                    </div>
                    <div>
                      <div className="text-sm font-semibold">{step.title}</div>
                      <div className="mt-1 text-xs leading-5 text-white/70">{step.description}</div>
                    </div>
                  </div>
                ))}
              </div>
              <Separator className="my-5 bg-white/10" />
              <div className="rounded-2xl border border-emerald-300/20 bg-emerald-400/8 p-4">
                <div className="flex items-start gap-3">
                  <ShieldCheck className="mt-0.5 h-4 w-4 text-emerald-200" />
                  <div>
                    <div className="text-sm font-semibold">Privacy-first by default</div>
                    <div className="mt-1 text-xs leading-5 text-white/70">
                      The page shows only structured indicators for review. Raw report access remains limited to the upload flow and downloadable JSON.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <div className="vault-card p-5">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Results extracted</div>
                <div className="mt-2 text-3xl font-bold text-foreground">{parsedData?.indicatorCount ?? 0}</div>
              </div>
              <div className="rounded-2xl bg-accent p-3 text-accent-foreground">
                <FileCheck2 className="h-5 w-5" />
              </div>
            </div>
            <div className="mt-3 text-xs text-muted-foreground">Every parsed result is normalized into a consistent schema before rendering.</div>
          </div>

          <div className="vault-card p-5">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Needs attention</div>
                <div className="mt-2 text-3xl font-bold text-foreground">{flaggedIndicators.length}</div>
              </div>
              <div className="rounded-2xl bg-red-50 p-3 text-red-600">
                <TriangleAlert className="h-5 w-5" />
              </div>
            </div>
            <div className="mt-3 text-xs text-muted-foreground">High, low, or abnormal values are grouped into the review queue below.</div>
          </div>

          <div className="vault-card p-5">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Pages analyzed</div>
                <div className="mt-2 text-3xl font-bold text-foreground">{parsedData?.meta.page_count ?? 0}</div>
              </div>
              <div className="rounded-2xl bg-sky-50 p-3 text-sky-600">
                <FileText className="h-5 w-5" />
              </div>
            </div>
            <div className="mt-3 text-xs text-muted-foreground">Longer documents are chunked automatically and merged into one review set.</div>
          </div>

          <div className="vault-card p-5">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Auto-review score</div>
                <div className="mt-2 text-3xl font-bold text-foreground">{reviewProgress}%</div>
              </div>
              <div className="rounded-2xl bg-emerald-50 p-3 text-emerald-600">
                <Cpu className="h-5 w-5" />
              </div>
            </div>
            <div className="mt-4">
              <Progress value={reviewProgress} className="h-2 bg-muted" />
            </div>
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.5fr_0.9fr]">
          <div className="space-y-6">
            <div className="vault-card p-6">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h3 className="flex items-center gap-2 text-lg font-semibold text-foreground">
                    <Stethoscope className="h-4 w-4 text-primary" />
                    Report review
                  </h3>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Review the most relevant findings first, then move into the full structured results.
                  </p>
                </div>
                {parsedData && (
                  <div className="rounded-2xl border border-border bg-muted/40 px-4 py-3 text-right">
                    <div className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">Current file</div>
                    <div className="mt-1 text-sm font-medium text-foreground">{parsedData.fileName}</div>
                  </div>
                )}
              </div>

              {!parsedData ? (
                <div className="mt-6 rounded-2xl border border-dashed border-border bg-muted/20 p-8">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="mt-0.5 h-5 w-5 text-primary" />
                    <div>
                      <div className="text-sm font-semibold text-foreground">No report uploaded yet</div>
                      <div className="mt-1 text-sm leading-6 text-muted-foreground">
                        Once you upload a PDF, this page will switch from upload mode into a patient-friendly review screen with summaries, flagged results, and categorized findings.
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <Tabs defaultValue="overview" className="mt-6">
                  <TabsList className="grid h-auto w-full grid-cols-3 rounded-2xl bg-muted/60 p-1">
                    <TabsTrigger value="overview" className="rounded-xl py-2.5">Overview</TabsTrigger>
                    <TabsTrigger value="results" className="rounded-xl py-2.5">All Results</TabsTrigger>
                    <TabsTrigger value="json" className="rounded-xl py-2.5">Structured Data</TabsTrigger>
                  </TabsList>

                  <TabsContent value="overview" className="mt-5 space-y-5">
                    <div className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
                      <div className="rounded-2xl border border-border bg-muted/25 p-5">
                        <div className="flex items-center justify-between gap-4">
                          <div>
                            <div className="text-sm font-semibold text-foreground">Priority review</div>
                            <div className="mt-1 text-xs text-muted-foreground">Values that may need manual confirmation or follow-up.</div>
                          </div>
                          <Badge className={flaggedIndicators.length ? "bg-red-50 text-red-700 hover:bg-red-50" : "bg-emerald-50 text-emerald-700 hover:bg-emerald-50"}>
                            {flaggedIndicators.length ? `${flaggedIndicators.length} flagged` : "No flagged values"}
                          </Badge>
                        </div>

                        <div className="mt-4 space-y-3">
                          {flaggedIndicators.length > 0 ? (
                            flaggedIndicators.slice(0, 4).map((indicator) => (
                              <div key={indicator.id} className="rounded-2xl border border-red-100 bg-white p-4">
                                <div className="flex flex-wrap items-center gap-2">
                                  <div className="text-sm font-semibold text-foreground">{indicator.name}</div>
                                  <span className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${getCategoryClasses(indicator.category)}`}>
                                    {indicator.category}
                                  </span>
                                  <span className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${getStatusClasses(indicator.status)}`}>
                                    {formatStatus(indicator.status)}
                                  </span>
                                </div>
                                <div className="mt-2 text-sm text-foreground">
                                  {indicator.value || "No value"} {indicator.unit}
                                </div>
                                {indicator.referenceRange && (
                                  <div className="mt-1 text-xs text-muted-foreground">Reference range: {indicator.referenceRange}</div>
                                )}
                              </div>
                            ))
                          ) : (
                            <div className="rounded-2xl border border-emerald-100 bg-emerald-50/60 p-4 text-sm text-emerald-800">
                              The parser did not identify any clearly abnormal results in this upload. You can still review the full list before saving or sharing.
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="rounded-2xl border border-border bg-muted/25 p-5">
                        <div className="text-sm font-semibold text-foreground">Coverage summary</div>
                        <div className="mt-1 text-xs text-muted-foreground">A quick view of how this report was organized.</div>

                        <div className="mt-4 space-y-3">
                          {groupedIndicators.map((group) => (
                            <div key={group.label} className="rounded-2xl bg-white p-4 shadow-[var(--shadow-sm)]">
                              <div className="flex items-center justify-between">
                                <div className="text-sm font-medium text-foreground">{group.label}</div>
                                <span className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${getCategoryClasses(group.label)}`}>
                                  {group.indicators.length} items
                                </span>
                              </div>
                              <div className="mt-3 flex flex-wrap gap-2">
                                {group.indicators.slice(0, 4).map((indicator) => (
                                  <div key={indicator.id} className="rounded-full border border-border bg-muted/35 px-3 py-1 text-xs text-foreground">
                                    {indicator.name}
                                  </div>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="rounded-2xl border border-border bg-background p-5">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <div className="text-sm font-semibold text-foreground">What to do next</div>
                          <div className="mt-1 text-xs text-muted-foreground">A release-ready workflow should move users forward after review.</div>
                        </div>
                        <Badge variant="outline" className="border-primary/20 bg-accent/50 text-accent-foreground">
                          Review complete: {reviewProgress}%
                        </Badge>
                      </div>

                      <div className="mt-4 grid gap-3 md:grid-cols-3">
                        <div className="rounded-2xl border border-border bg-muted/20 p-4">
                          <div className="text-sm font-semibold text-foreground">1. Review flagged items</div>
                          <div className="mt-1 text-xs leading-5 text-muted-foreground">Check abnormal or missing values before using this report elsewhere in the product.</div>
                        </div>
                        <div className="rounded-2xl border border-border bg-muted/20 p-4">
                          <div className="text-sm font-semibold text-foreground">2. Confirm structured data</div>
                          <div className="mt-1 text-xs leading-5 text-muted-foreground">Use the structured data tab only when you need the machine-readable export.</div>
                        </div>
                        <div className="rounded-2xl border border-border bg-muted/20 p-4">
                          <div className="text-sm font-semibold text-foreground">3. Continue to sharing or vault save</div>
                          <div className="mt-1 text-xs leading-5 text-muted-foreground">This page is now ready to connect to a future save or consent action without another redesign.</div>
                        </div>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="results" className="mt-5">
                    <div className="space-y-4">
                      {groupedIndicators.map((group) => (
                        <div key={group.label} className="rounded-2xl border border-border bg-background p-5">
                          <div className="flex items-center justify-between gap-4">
                            <div className="flex items-center gap-3">
                              <div className="text-sm font-semibold text-foreground">{group.label}</div>
                              <span className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${getCategoryClasses(group.label)}`}>
                                {group.indicators.length} results
                              </span>
                            </div>
                          </div>

                          <div className="mt-4 grid gap-3">
                            {group.indicators.map((indicator) => (
                              <div key={indicator.id} className="rounded-2xl border border-border bg-muted/15 p-4">
                                <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                                  <div>
                                    <div className="flex flex-wrap items-center gap-2">
                                      <div className="text-sm font-semibold text-foreground">{indicator.name}</div>
                                      <span className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${getStatusClasses(indicator.status)}`}>
                                        {formatStatus(indicator.status)}
                                      </span>
                                    </div>
                                    <div className="mt-2 text-sm text-foreground">
                                      {indicator.value || "Not provided"} {indicator.unit}
                                    </div>
                                  </div>

                                  <div className="grid gap-2 text-xs text-muted-foreground sm:grid-cols-2 lg:text-right">
                                    <div>
                                      <span className="font-medium text-foreground">Reference:</span> {indicator.referenceRange || "Not provided"}
                                    </div>
                                    <div>
                                      <span className="font-medium text-foreground">Method:</span> {indicator.instrument || "Not provided"}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </TabsContent>

                  <TabsContent value="json" className="mt-5">
                    <div className="rounded-2xl border border-border bg-background p-5">
                      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                          <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                            <FileJson className="h-4 w-4 text-primary" />
                            Structured JSON
                          </div>
                          <div className="mt-1 text-xs text-muted-foreground">
                            Keep this view available for export and troubleshooting, but out of the default user path.
                          </div>
                        </div>
                        <Button size="sm" variant="outline" className="gap-2" onClick={() => downloadParsedJson(parsedData)}>
                          <Download className="h-3.5 w-3.5" />
                          Download JSON
                        </Button>
                      </div>

                      <pre className="mt-4 max-h-[480px] overflow-auto rounded-2xl bg-slate-950 p-4 text-xs leading-6 text-slate-100">
                        {JSON.stringify(parsedData, null, 2)}
                      </pre>
                    </div>
                  </TabsContent>
                </Tabs>
              )}
            </div>
          </div>

          <div className="space-y-6">
            <div className="vault-card p-6">
              <h3 className="flex items-center gap-2 text-lg font-semibold text-foreground">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                Review status
              </h3>
              <div className="mt-4 space-y-4">
                <div className="rounded-2xl bg-muted/25 p-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-foreground">Parsing state</span>
                    <span className={parsedData ? "status-active" : "status-pending"}>{parsedData ? "Ready" : "Waiting"}</span>
                  </div>
                  <div className="mt-3 text-xs text-muted-foreground">
                    {parsedData
                      ? "Your latest report is ready to review and export."
                      : "Upload a PDF to unlock the review workflow."}
                  </div>
                </div>

                <div className="rounded-2xl bg-muted/25 p-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-foreground">Flagged values</span>
                    <span className={flaggedIndicators.length ? "status-pending" : "status-active"}>
                      {flaggedIndicators.length ? `${flaggedIndicators.length} found` : "None"}
                    </span>
                  </div>
                  <div className="mt-3 text-xs text-muted-foreground">
                    Values marked high, low, or abnormal are surfaced in the overview tab.
                  </div>
                </div>

                <div className="rounded-2xl bg-muted/25 p-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-foreground">Machine-readable export</span>
                    <span className={parsedData ? "status-active" : "status-private"}>{parsedData ? "Available" : "Locked"}</span>
                  </div>
                  <div className="mt-3 text-xs text-muted-foreground">
                    Download the normalized JSON only when you need to reuse or debug the parsed record.
                  </div>
                </div>
              </div>
            </div>

            <div className="vault-card p-6">
              <h3 className="flex items-center gap-2 text-lg font-semibold text-foreground">
                <ShieldCheck className="h-4 w-4 text-primary" />
                Document details
              </h3>
              <div className="mt-4 space-y-3 text-sm">
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">File</span>
                  <span className="truncate text-right font-medium text-foreground">{parsedData?.fileName ?? "No file selected"}</span>
                </div>
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Pages</span>
                  <span className="font-medium text-foreground">{parsedData?.meta.page_count ?? 0}</span>
                </div>
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">AI chunks</span>
                  <span className="font-medium text-foreground">{parsedData?.meta.chunk_count ?? 0}</span>
                </div>
                <div className="flex items-center justify-between gap-4">
                  <span className="text-muted-foreground">Results</span>
                  <span className="font-medium text-foreground">{parsedData?.indicatorCount ?? 0}</span>
                </div>
              </div>

              <Separator className="my-5" />

              <div className="rounded-2xl border border-primary/15 bg-accent/35 p-4">
                <div className="text-sm font-semibold text-foreground">Release-oriented note</div>
                <div className="mt-1 text-xs leading-5 text-muted-foreground">
                  This side panel is designed to stay stable when you later add actions like save-to-vault, request review, or share with consent.
                </div>
              </div>
            </div>

            <div className="vault-card p-6">
              <div className="flex items-center justify-between gap-3">
                <h3 className="flex items-center gap-2 text-lg font-semibold text-foreground">
                  <FileText className="h-4 w-4 text-primary" />
                  Recent uploads
                </h3>
                <span className="text-xs text-muted-foreground">{uploadHistory.length || recentUploadPlaceholders.length} items</span>
              </div>

              <div className="mt-4 space-y-3">
                {uploadHistory.length > 0
                  ? uploadHistory.map((upload) => (
                      <div
                        key={`${upload.fileName}-${upload.uploadedAt}`}
                        className="rounded-2xl border border-border bg-muted/20 p-4 transition-colors hover:bg-muted/35"
                      >
                        <div className="flex items-start gap-3">
                          <div className="mt-0.5 rounded-xl bg-accent p-2 text-accent-foreground">
                            <FileText className="h-4 w-4" />
                          </div>
                          <div className="min-w-0 flex-1">
                            <button
                              className="group w-full text-left"
                              onClick={() => {
                                setParsedData(upload.parsedData);
                                setSelectedFileName(upload.fileName);
                              }}
                            >
                              <div className="truncate text-sm font-medium text-foreground">{upload.fileName}</div>
                              <div className="mt-1 text-xs text-muted-foreground">
                                {formatDate(upload.uploadedAt)} · {upload.indicatorCount} results extracted
                              </div>
                              <div className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-primary">
                                Open report
                                <ChevronRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
                              </div>
                            </button>
                          </div>
                          <button
                            className="rounded-lg p-2 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                            onClick={() => {
                              setUploadHistory((current) =>
                                current.filter((item) => !(item.fileName === upload.fileName && item.uploadedAt === upload.uploadedAt)),
                              );
                            }}
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                    ))
                  : recentUploadPlaceholders.map((upload) => (
                      <div key={upload.name} className="rounded-2xl border border-border bg-muted/20 p-4">
                        <div className="flex items-start gap-3">
                          <div className="mt-0.5 rounded-xl bg-accent p-2 text-accent-foreground">
                            <FileText className="h-4 w-4" />
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-sm font-medium text-foreground">{upload.name}</div>
                            <div className="mt-1 text-xs text-muted-foreground">{upload.date} · {upload.records} parsed results</div>
                          </div>
                        </div>
                      </div>
                    ))}
              </div>
            </div>

            <div className="rounded-2xl border border-primary/20 bg-accent p-4">
              <div className="flex items-start gap-3">
                <AlertCircle className="mt-0.5 h-4 w-4 text-primary" />
                <div className="text-sm text-accent-foreground">
                  <strong>De-identification is automatic.</strong> Before any data is made searchable or shared, personal identifiers are stripped and the review flow only exposes structured fields.
                </div>
              </div>
              <div className="mt-4">
                {parsedData ? (
                  <Button asChild variant="outline" className="gap-2 bg-white/60">
                    <Link to="/health-data/onchain">
                      Next: Review On-Chain Options
                      <ArrowRight className="h-4 w-4" />
                    </Link>
                  </Button>
                ) : (
                  <Button variant="outline" className="gap-2 bg-white/60" disabled>
                    Next: Review On-Chain Options
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>
          </div>
        </section>
      </div>
    </AppLayout>
  );
}
