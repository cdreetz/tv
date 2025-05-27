"use client"

import { SidebarLeft } from "@/components/sidebar-left"
import { SidebarRight } from "@/components/sidebar-right"
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb"
import { Separator } from "@/components/ui/separator"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { useForm } from "react-hook-form"
import { z } from "zod"
import { zodResolver } from "@hookform/resolvers/zod"
import { Button } from "@/components/ui/button"
import { X, Loader2, CheckCircle, XCircle } from "lucide-react"
import { useState, useEffect, useRef } from "react"

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

export default function Page() {
  const [companies, setCompanies] = useState<Company[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentStatus, setCurrentStatus] = useState<PipelineStatus | null>(null)
  const [statuses, setStatuses] = useState<PipelineStatus[]>([])
  const [generatedReport, setGeneratedReport] = useState<ReportData | string | null>(null)
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
        setStatuses(prev => [...prev, status])

        if (status.stage === "completion" && status.data) {
          setGeneratedReport(status.data)
          setIsGenerating(false)
          eventSource.close()
        } else if (status.stage === "error") {
          setIsGenerating(false)
          eventSource.close()
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

  const getStageIcon = (stage: string) => {
    switch (stage) {
      case "completion":
        return <CheckCircle className="h-4 w-4 text-green-600" />
      case "data_enrichment_complete":
        return <CheckCircle className="h-4 w-4 text-blue-600" />
      case "error":
        return <XCircle className="h-4 w-4 text-red-600" />
      default:
        return <Loader2 className="h-4 w-4 text-blue-600 animate-spin" />
    }
  }

  return (
    <SidebarProvider>
      <SidebarLeft />
      <SidebarInset>
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
                    {isGenerating ? (
                      <>
                        <Loader2 className="h-5 w-5 animate-spin" />
                        Generating Report
                      </>
                    ) : (
                      <>
                        <CheckCircle className="h-5 w-5 text-green-600" />
                        Report Generation Complete
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
                    {getStageIcon(currentStatus.stage)}
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
                            {getStageIcon(status.stage)}
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
      </SidebarInset>
      <SidebarRight />
    </SidebarProvider>
  )
}
