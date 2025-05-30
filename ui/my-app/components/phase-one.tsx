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
import { CheckCircle, XCircle, Loader2, X } from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Form, FormField, FormItem, FormLabel, FormControl, FormMessage } from "./ui/form";

interface Company {
  id: string
  name: string
}

interface ReportData {
  company_summary?: string
  structured_data?: any
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


export default function PhaseOne() {
  const [companies, setCompanies] = useState<Company[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentStatus, setCurrentStatus] = useState<PipelineStatus | null>(null)
  const [statuses, setStatuses] = useState<PipelineStatus[]>([])
  const [generatedReport, setGeneratedReport] = useState<ReportData | string | null>(null)
  const [isCompleted, setIsCompleted] = useState(false)
  const eventSourceRef = useRef<EventSource | null>(null)
  
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      company: "",
    },
  })
 
  function onSubmit(values: z.infer<typeof formSchema>) {
    const newCompany: Company = {
      id: Date.now().toString(),
      name: values.company
    }
    setCompanies(prev => [...prev, newCompany])
    form.reset()
  }

  function removeCompany(id: string) {
    setCompanies(prev => prev.filter(company => company.id !== id))
  }

  function handleGenerateReport() {
    if (companies.length === 0) return
    
    setIsGenerating(true)
    setGeneratedReport(null)
    setIsCompleted(false)
    // Don't reset statuses and currentStatus - let them persist
    // setStatuses([])
    // setCurrentStatus(null)

    // Determine which endpoint to use
    const endpoint = companies.length === 1 
      ? `/sse/report/${encodeURIComponent(companies[0].name)}`
      : `/sse/comparison?companies=${companies.map(c => encodeURIComponent(c.name)).join(',')}`

    const eventSource = new EventSource(`http://localhost:8000${endpoint}`)
    eventSourceRef.current = eventSource

    eventSource.onmessage = (event) => {
      try {
        const status: PipelineStatus = JSON.parse(event.data)
        setCurrentStatus(status)
        
        if (status.stage === "completion") {
          // When completion is reached, mark all previous stages as complete
          setStatuses(prev => {
            const updatedStatuses = [...prev]
            
            // Add the completion status
            updatedStatuses.push(status)
            
            // Create a map of stages we've seen
            const stageMap = new Map<string, PipelineStatus>()
            updatedStatuses.forEach(s => {
              stageMap.set(s.stage, s)
            })
            
            // Define the typical pipeline stages in order
            const expectedStages = [
              "initialization",
              "validation", 
              "data_enrichment",
              "data_enrichment_complete",
              "data_extraction",
              "similarity_analysis", 
              "report_generation",
              "completion"
            ]
            
            // For any stage that exists but isn't completion, mark it as complete
            const finalStatuses: PipelineStatus[] = []
            expectedStages.forEach(stageName => {
              if (stageMap.has(stageName)) {
                const existingStatus = stageMap.get(stageName)!
                if (stageName !== "completion" && existingStatus.stage !== "completion") {
                  // Mark this stage as complete
                  finalStatuses.push({
                    ...existingStatus,
                    stage: existingStatus.stage,
                    message: existingStatus.message + " ✓",
                    progress: 1.0
                  })
                } else {
                  finalStatuses.push(existingStatus)
                }
              }
            })
            
            // Add any other stages that weren't in our expected list
            updatedStatuses.forEach(s => {
              if (!expectedStages.includes(s.stage)) {
                if (s.stage !== "completion") {
                  finalStatuses.push({
                    ...s,
                    message: s.message + " ✓",
                    progress: 1.0
                  })
                } else {
                  finalStatuses.push(s)
                }
              }
            })
            
            return finalStatuses
          })
          
          setGeneratedReport(status.data || null)
          setIsGenerating(false)
          setIsCompleted(true)
          eventSource.close()
        } else if (status.stage === "error") {
          setStatuses(prev => [...prev, status])
          setIsGenerating(false)
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
      setIsGenerating(false)
      eventSource.close()
    }
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
                    Market Report Pipeline
                  </BreadcrumbPage>
                </BreadcrumbItem>
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </header>
        <div className="flex flex-1 flex-col gap-4 p-4">
          <div className="bg-muted/50 mx-auto w-full max-w-3xl rounded-xl" >
            <Card>
              <CardHeader>
                <CardTitle>Market Report Pipeline</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-col gap-4 h-full">
                  <Form {...form}>
                    <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
                      <FormField
                      control={form.control}
                      name="company"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Company</FormLabel>
                          <div className="flex flex-row gap-4">
                            <FormControl>
                              <Input {...field} />
                            </FormControl>
                            <Button type="submit" className="bg-green-600 hover:bg-green-700 text-white">Add Company</Button>
                          </div>
                          <FormMessage />
                        </FormItem>
                      )}
                      />
                    </form>
                  </Form>

                  <div className="flex flex-col gap-2">
                    <h3 className="font-medium">Selected Companies</h3>
                    <div className="flex flex-wrap gap-2">
                      {companies.map((company) => (
                        <div key={company.id} className="bg-secondary text-secondary-foreground px-3 py-1 rounded-md flex items-center gap-2">
                          {company.name}
                          <Button 
                            variant="ghost" 
                            size="sm" 
                            className="h-auto p-0 hover:bg-transparent"
                            onClick={() => removeCompany(company.id)}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>

                  {companies.length > 0 && (
                    <div className="flex justify-center pt-4">
                      <Button 
                        className="bg-blue-600 hover:bg-blue-700 text-white px-8"
                        onClick={handleGenerateReport}
                        disabled={isGenerating}
                      >
                        {isGenerating ? 'Generating...' : 'Generate Market Report'}
                      </Button>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Streaming Status */}
          {currentStatus && statuses.length > 0 && (
            <div className="bg-muted/50 mx-auto w-full max-w-3xl rounded-xl">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    {isCompleted ? (
                      <>
                        <CheckCircle className="h-5 w-5 text-green-600" />
                        Report Generation Complete
                      </>
                    ) : (
                      <>
                        <Loader2 className="h-5 w-5 animate-spin" />
                        Generating Report
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
                        className="bg-primary h-2 rounded-full transition-all duration-300"
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

                  {/* Status History */}
                  {statuses.length > 1 && (
                    <div className="space-y-2">
                      <h4 className="text-sm font-medium">Recent Updates</h4>
                      <div className="max-h-32 overflow-y-auto space-y-1">
                        {statuses.slice(-5).reverse().map((status, index) => (
                          <div key={index} className="flex items-center gap-2 text-xs p-2 rounded bg-muted/30">
                            {getStageIcon(status.stage, isCompleted && status.stage !== "completion")}
                            <span className="font-medium">{status.stage.replace(/_/g, ' ')}</span>
                            <span className="flex-1 truncate">{status.message}</span>
                            <span className="text-muted-foreground">
                              {Math.round(status.progress * 100)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          )}

          {/* Generated Report */}
          {generatedReport && (
            <div className="bg-muted/50 mx-auto w-full max-w-3xl rounded-xl">
              <Card>
                <CardHeader>
                  <CardTitle>Generated Report</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="prose prose-sm max-w-none dark:prose-invert">
                    <div 
                      className="whitespace-pre-wrap text-sm bg-background p-6 rounded-lg border overflow-auto max-h-96"
                      style={{ fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace' }}
                    >
                      {typeof generatedReport === 'string' 
                        ? generatedReport 
                        : generatedReport?.company_summary || 
                          (generatedReport?.structured_data ? JSON.stringify(generatedReport.structured_data, null, 2) : 
                           'No report data available')}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          <div className="bg-muted/50 mx-auto h-[100vh] w-full max-w-3xl rounded-xl" />
        </div>
      </div>
    )
}