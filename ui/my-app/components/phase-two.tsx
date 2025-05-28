'use client'

import { SidebarTrigger } from "./ui/sidebar";
import { CardContent, CardHeader, CardTitle } from "./ui/card";
import { Card } from "./ui/card";
import { Separator } from "./ui/separator";
import { Breadcrumb, BreadcrumbList, BreadcrumbItem, BreadcrumbPage } from "./ui/breadcrumb";
import { useEffect, useRef } from "react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { CheckCircle, XCircle, Loader2, Building2, TrendingUp } from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Form, FormField, FormItem, FormLabel, FormControl, FormMessage } from "./ui/form";

interface SimilarCompany {
  name: string
  similarity_score: number
  raw_score?: number
  summary: string
  matched_terms?: string[]
  match_count?: number
  match_details?: string[]
  similarity_reason?: string
  key_differentiator?: string
  other_product_offering?: string
  claude_similarity_reason?: string
  claude_key_differentiator?: string
}

interface ReportData {
  company_summary?: string
  structured_data?: any
  similar_companies?: SimilarCompany[]
  market_categories?: string[]
  key_terms?: string[]
  search_terms?: string[]
  product_offering_short?: string
}

interface PipelineStatus {
  stage: string
  message: string
  progress: number
  data?: ReportData | string
  error?: string
  timestamp: string
}

const formSchema = z.object({
  company: z.string().min(2, {
    message: "Company must be at least 2 characters.",
  }),
})

export default function PhaseTwo() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [currentStatus, setCurrentStatus] = useState<PipelineStatus | null>(null)
  const [statuses, setStatuses] = useState<PipelineStatus[]>([])
  const [analysisResult, setAnalysisResult] = useState<ReportData | null>(null)
  const [isCompleted, setIsCompleted] = useState(false)
  const [selectedCompany, setSelectedCompany] = useState<string>("")
  const eventSourceRef = useRef<EventSource | null>(null)
  
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      company: "",
    },
  })
 
  function onSubmit(values: z.infer<typeof formSchema>) {
    setSelectedCompany(values.company)
    handleAnalyzeCompany(values.company)
  }

  function handleAnalyzeCompany(companyName: string) {
    setIsAnalyzing(true)
    setAnalysisResult(null)
    setIsCompleted(false)
    setStatuses([])
    setCurrentStatus(null)

    const endpoint = `/sse/report/${encodeURIComponent(companyName)}`
    const eventSource = new EventSource(`http://localhost:8000${endpoint}`)
    eventSourceRef.current = eventSource

    eventSource.onmessage = (event) => {
      try {
        const status: PipelineStatus = JSON.parse(event.data)
        setCurrentStatus(status)
        
        if (status.stage === "completion") {
          // Parse the report data to extract similar companies
          let reportData: ReportData = {}
          
          if (typeof status.data === 'string') {
            // Extract similar companies from the markdown report
            const reportText = status.data
            const marketData = extractMarketData(reportText)
            reportData.company_summary = extractCompanySummary(reportText)
            reportData.similar_companies = extractSimilarCompanies(reportText)
            reportData.market_categories = marketData.market_categories
            reportData.key_terms = marketData.key_terms
            reportData.search_terms = marketData.search_terms
            reportData.product_offering_short = marketData.product_offering_short
          } else if (status.data && typeof status.data === 'object') {
            // Handle structured data from backend
            const structuredData = status.data as any
            reportData = {
              company_summary: structuredData.company_summary,
              similar_companies: structuredData.similar_companies,
              market_categories: structuredData.market_categories,
              key_terms: structuredData.key_terms,
              search_terms: structuredData.search_terms,
              product_offering_short: structuredData.product_offering_short,
              structured_data: structuredData.structured_data
            }
          }
          
          setAnalysisResult(reportData)
          setIsAnalyzing(false)
          setIsCompleted(true)
          eventSource.close()
        } else if (status.stage === "error") {
          setStatuses(prev => [...prev, status])
          setIsAnalyzing(false)
          eventSource.close()
        } else {
          setStatuses(prev => [...prev, status])
        }
      } catch (error) {
        console.error('Error parsing SSE data:', error)
      }
    }

    eventSource.onerror = (error) => {
      console.error('SSE error:', error)
      setIsAnalyzing(false)
      eventSource.close()
    }
  }

  // Helper function to extract company summary from markdown report
  function extractCompanySummary(reportText: string): string {
    const summaryMatch = reportText.match(/## Company Overview[\s\S]*?\n([\s\S]*?)(?=\n##|$)/)
    return summaryMatch ? summaryMatch[1].trim() : ""
  }

  // Helper function to extract similar companies from markdown report
  function extractSimilarCompanies(reportText: string): SimilarCompany[] {
    const competitiveSection = reportText.match(/## Competitive Landscape[\s\S]*?\n\n([\s\S]*?)(?=\n##|$)/)
    if (!competitiveSection) return []
    
    const companiesText = competitiveSection[1]
    
    // Split by company sections and parse each one
    const companySections = companiesText.split(/(?=### \d+\.)/).filter(section => section.trim())
    
    const companies: SimilarCompany[] = []
    for (const section of companySections) {
      const nameMatch = section.match(/### \d+\.\s*(.*?)\s*\n/)
      const scoreMatch = section.match(/\*\*Similarity Score:\*\*\s*([\d.]+)/)
      const rawScoreMatch = section.match(/\*\*Raw Match Score:\*\*\s*([\d.]+)/)
      const matchedTermsMatch = section.match(/\*\*Matched Terms:\*\*\s*(.*?)\n/)
      const otherProductMatch = section.match(/\*\*Their Product Focus:\*\*\s*(.*?)\n/)
      const similarityReasonMatch = section.match(/\*\*Similarity Reason:\*\*\s*(.*?)\n/)
      const howSimilarMatch = section.match(/\*\*How They're Similar:\*\*\s*(.*?)\n/)
      const keyDifferentiatorMatch = section.match(/\*\*Key Differentiator:\*\*\s*(.*?)\n/)
      const briefMatch = section.match(/\*\*Brief:\*\*\s*([\s\S]*?)(?=\n###|$)/)
      
      if (nameMatch && scoreMatch && briefMatch) {
        const company: SimilarCompany = {
          name: nameMatch[1].trim(),
          similarity_score: parseFloat(scoreMatch[1]),
          summary: briefMatch[1].trim().replace(/\.\.\.$/, '')
        }
        
        // Add raw score if available
        if (rawScoreMatch) {
          company.raw_score = parseFloat(rawScoreMatch[1])
        }
        
        // Add matched terms if available
        if (matchedTermsMatch) {
          company.matched_terms = matchedTermsMatch[1].split(',').map(term => term.trim())
          company.match_count = company.matched_terms.length
        }
        
        // Add other product offering if available
        if (otherProductMatch) {
          company.other_product_offering = otherProductMatch[1].trim()
        }
        
        // Prioritize Claude-generated similarity reason
        if (howSimilarMatch) {
          company.claude_similarity_reason = howSimilarMatch[1].trim()
        } else if (similarityReasonMatch) {
          company.similarity_reason = similarityReasonMatch[1].trim()
        }
        
        // Add key differentiator (Claude-generated is already prioritized in backend)
        if (keyDifferentiatorMatch) {
          const differentiator = keyDifferentiatorMatch[1].trim()
          // Check if this looks like a Claude-generated response (more sophisticated)
          if (differentiator.length > 50 && (differentiator.includes('focuses on') || differentiator.includes('specializes in') || differentiator.includes('differentiates itself'))) {
            company.claude_key_differentiator = differentiator
          } else {
            company.key_differentiator = differentiator
          }
        }
        
        companies.push(company)
      }
    }
    
    return companies
  }

  // Helper function to extract market categories and key terms
  function extractMarketData(reportText: string): { market_categories: string[], key_terms: string[], search_terms: string[], product_offering_short: string } {
    const marketCategoriesMatch = reportText.match(/\*\*Market Categories:\*\*\s*(.*?)\n/)
    const keyTermsMatch = reportText.match(/\*\*Key Terms:\*\*\s*(.*?)\n/)
    const searchTermsMatch = reportText.match(/\*\*Search Terms:\*\*\s*(.*?)\n/)
    const productFocusMatch = reportText.match(/\*\*Product Focus:\*\*\s*(.*?)\n/)
    
    const market_categories = marketCategoriesMatch 
      ? marketCategoriesMatch[1].split(',').map(cat => cat.trim()).filter(cat => cat && cat !== 'General technology')
      : []
    
    const key_terms = keyTermsMatch 
      ? keyTermsMatch[1].split(',').map(term => term.trim()).filter(term => term && term !== 'Not specified')
      : []
    
    const search_terms = searchTermsMatch 
      ? searchTermsMatch[1].split(',').map(term => term.trim()).filter(term => term && term !== 'Not specified')
      : []
    
    const product_offering_short = productFocusMatch 
      ? productFocusMatch[1].trim()
      : ""
    
    return { market_categories, key_terms, search_terms, product_offering_short }
  }

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
      }
    }
  }, [])

  const getStageIcon = (stage: string, isCompleted: boolean = false) => {
    if (isCompleted || stage === "completion" || stage === "data_enrichment_complete") {
      return <CheckCircle className="h-4 w-4 text-green-600" />
    }
    
    switch (stage) {
      case "error":
        return <XCircle className="h-4 w-4 text-red-600" />
      default:
        return <Loader2 className="h-4 w-4 text-blue-600 animate-spin" />
    }
  }

  const getSimilarityColor = (score: number) => {
    if (score >= 0.7) return "bg-green-100 text-green-800 border-green-200"
    if (score >= 0.5) return "bg-yellow-100 text-yellow-800 border-yellow-200"
    return "bg-red-100 text-red-800 border-red-200"
  }

  return (
    <div className="flex flex-1 flex-col">
      <header className="bg-background sticky top-0 flex h-14 shrink-0 items-center gap-2">
        <div className="flex flex-1 items-center gap-2 px-3">
          <SidebarTrigger />
          <Separator
            orientation="vertical"
            className="mr-2 data-[orientation=vertical]:h-4"
          />
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem>
                <BreadcrumbPage className="line-clamp-1">
                  Similar Companies Finder
                </BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </div>
      </header>
      
      <div className="flex flex-1 flex-col gap-4 p-4">
        <div className="bg-muted/50 mx-auto w-full max-w-4xl rounded-xl">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Building2 className="h-5 w-5" />
                Find Similar Companies
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                Enter a company name to discover similar companies in the market using keyword-based similarity analysis with market categories and key terms.
              </p>
            </CardHeader>
            <CardContent>
              <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                  <FormField
                    control={form.control}
                    name="company"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Company Name</FormLabel>
                        <div className="flex flex-row gap-4">
                          <FormControl>
                            <Input 
                              {...field} 
                              placeholder="e.g., OpenAI, Stripe, Airbnb"
                              className="flex-1"
                            />
                          </FormControl>
                          <Button 
                            type="submit" 
                            className="bg-purple-600 hover:bg-purple-700 text-white"
                            disabled={isAnalyzing}
                          >
                            {isAnalyzing ? (
                              <>
                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                Analyzing...
                              </>
                            ) : (
                              <>
                                <TrendingUp className="h-4 w-4 mr-2" />
                                Find Similar
                              </>
                            )}
                          </Button>
                        </div>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </form>
              </Form>
            </CardContent>
          </Card>
        </div>

        {/* Analysis Progress */}
        {currentStatus && statuses.length > 0 && (
          <div className="bg-muted/50 mx-auto w-full max-w-4xl rounded-xl">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {isCompleted ? (
                    <>
                      <CheckCircle className="h-5 w-5 text-green-600" />
                      Analysis Complete
                    </>
                  ) : (
                    <>
                      <Loader2 className="h-5 w-5 animate-spin" />
                      Analyzing {selectedCompany}
                    </>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Progress Bar */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Progress</span>
                    <span className="text-sm text-muted-foreground">
                      {Math.round(currentStatus.progress * 100)}%
                    </span>
                  </div>
                  <div className="w-full bg-secondary rounded-full h-2">
                    <div 
                      className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${currentStatus.progress * 100}%` }}
                    />
                  </div>
                </div>

                {/* Current Status */}
                <div className="flex items-center gap-3 p-3 rounded-lg border bg-muted/50">
                  {getStageIcon(currentStatus.stage, isCompleted)}
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">
                        {currentStatus.stage.replace(/_/g, ' ').toUpperCase()}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {new Date(currentStatus.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-sm mt-1">{currentStatus.message}</p>
                    {currentStatus.error && (
                      <p className="text-sm text-red-600 mt-1">Error: {currentStatus.error}</p>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Similar Companies Results */}
        {analysisResult && analysisResult.similar_companies && (
          <div className="bg-muted/50 mx-auto w-full max-w-4xl rounded-xl">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-purple-600" />
                  Similar Companies to {selectedCompany}
                </CardTitle>
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">
                    Found {analysisResult.similar_companies.length} similar companies using enhanced matching
                  </p>
                  {(analysisResult.product_offering_short) && (
                    <div className="flex flex-wrap gap-1">
                      <span className="text-xs font-medium text-muted-foreground">Product Focus:</span>
                      <span className="inline-flex items-center rounded-full bg-purple-100 px-2 py-1 text-xs font-medium text-purple-800">
                        {analysisResult.product_offering_short}
                      </span>
                    </div>
                  )}
                  {(analysisResult.market_categories && analysisResult.market_categories.length > 0) && (
                    <div className="flex flex-wrap gap-1">
                      <span className="text-xs font-medium text-muted-foreground">Market Categories:</span>
                      {analysisResult.market_categories.map((category, idx) => (
                        <span key={idx} className="inline-flex items-center rounded-full bg-blue-100 px-2 py-1 text-xs font-medium text-blue-800">
                          {category}
                        </span>
                      ))}
                    </div>
                  )}
                  {(analysisResult.key_terms && analysisResult.key_terms.length > 0) && (
                    <div className="flex flex-wrap gap-1">
                      <span className="text-xs font-medium text-muted-foreground">Key Terms:</span>
                      {analysisResult.key_terms.map((term, idx) => (
                        <span key={idx} className="inline-flex items-center rounded-full bg-green-100 px-2 py-1 text-xs font-medium text-green-800">
                          {term}
                        </span>
                      ))}
                    </div>
                  )}
                  {(analysisResult.search_terms && analysisResult.search_terms.length > 0) && (
                    <div className="flex flex-wrap gap-1">
                      <span className="text-xs font-medium text-muted-foreground">Search Terms:</span>
                      {analysisResult.search_terms.map((term, idx) => (
                        <span key={idx} className="inline-flex items-center rounded-full bg-yellow-100 px-2 py-1 text-xs font-medium text-yellow-800">
                          {term}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4">
                  {analysisResult.similar_companies.map((company, index) => (
                    <div key={index} className="border rounded-lg p-4 hover:bg-muted/50 transition-colors">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-purple-100 text-purple-600 font-semibold text-sm">
                            {index + 1}
                          </div>
                          <div>
                            <h3 className="font-semibold text-lg">{company.name}</h3>
                            <div className="flex items-center gap-2 mt-1">
                              <div className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold ${getSimilarityColor(company.similarity_score)}`}>
                                {(company.similarity_score * 100).toFixed(1)}% similarity
                              </div>
                              {company.raw_score && (
                                <div className="inline-flex items-center rounded-full bg-indigo-100 px-2.5 py-0.5 text-xs font-semibold text-indigo-800">
                                  {company.raw_score.toFixed(2)} raw score
                                </div>
                              )}
                              {company.match_count && (
                                <div className="inline-flex items-center rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-semibold text-gray-800">
                                  {company.match_count} matches
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Show matched terms if available */}
                      {company.matched_terms && company.matched_terms.length > 0 && (
                        <div className="mb-3">
                          <div className="flex flex-wrap gap-1">
                            <span className="text-xs font-medium text-muted-foreground">Matched Terms:</span>
                            {company.matched_terms.map((term, idx) => (
                              <span key={idx} className="inline-flex items-center rounded-full bg-orange-100 px-2 py-1 text-xs font-medium text-orange-800">
                                {term}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {/* Show other company's product offering if available */}
                      {company.other_product_offering && (
                        <div className="mb-3">
                          <div className="flex flex-wrap gap-1">
                            <span className="text-xs font-medium text-muted-foreground">Their Product Focus:</span>
                            <span className="inline-flex items-center rounded-full bg-purple-100 px-2 py-1 text-xs font-medium text-purple-800">
                              {company.other_product_offering}
                            </span>
                          </div>
                        </div>
                      )}
                      
                      {/* Similarity Reason - Prioritize Claude-generated */}
                      {(company.claude_similarity_reason || company.similarity_reason) && (
                        <div className="mb-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
                          <div className="flex items-start gap-2">
                            <div className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-100 flex items-center justify-center mt-0.5">
                              <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                            </div>
                            <div>
                              <div className="flex items-center gap-2 mb-1">
                                <h4 className="text-sm font-medium text-blue-900">How it's similar to {selectedCompany}</h4>
                                {company.claude_similarity_reason && (
                                  <span className="inline-flex items-center rounded-full bg-blue-200 px-1.5 py-0.5 text-xs font-medium text-blue-800">
                                    AI Analysis
                                  </span>
                                )}
                              </div>
                              <p className="text-sm text-blue-800">
                                {company.claude_similarity_reason || company.similarity_reason}
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {/* Key Differentiator - Prioritize Claude-generated */}
                      {(company.claude_key_differentiator || company.key_differentiator) && (
                        <div className="p-3 bg-green-50 rounded-lg border border-green-200">
                          <div className="flex items-start gap-2">
                            <div className="flex-shrink-0 w-5 h-5 rounded-full bg-green-100 flex items-center justify-center mt-0.5">
                              <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                            </div>
                            <div>
                              <div className="flex items-center gap-2 mb-1">
                                <h4 className="text-sm font-medium text-green-900">Key Differentiator</h4>
                                {company.claude_key_differentiator && (
                                  <span className="inline-flex items-center rounded-full bg-green-200 px-1.5 py-0.5 text-xs font-medium text-green-800">
                                    AI Analysis
                                  </span>
                                )}
                              </div>
                              <p className="text-sm text-green-800">
                                {company.claude_key_differentiator || company.key_differentiator}
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Company Overview */}
        {analysisResult && analysisResult.company_summary && (
          <div className="bg-muted/50 mx-auto w-full max-w-4xl rounded-xl">
            <Card>
              <CardHeader>
                <CardTitle>Company Overview: {selectedCompany}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="prose prose-sm max-w-none dark:prose-invert">
                  <p className="text-sm leading-relaxed">{analysisResult.company_summary}</p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        <div className="bg-muted/50 mx-auto h-[100vh] w-full max-w-4xl rounded-xl" />
      </div>
    </div>
  )
} 